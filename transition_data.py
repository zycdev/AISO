from html import unescape
import json
import logging
import random
import re
from typing import Callable, List

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import ElectraTokenizerFast  # , PreTrainedTokenizer, PreTrainedTokenizerFast

from basic_tokenizer import SimpleTokenizer
from config import ADDITIONAL_SPECIAL_TOKENS, FUNC2ID, NA_POS
from utils.data_utils import get_valid_links
from utils.tensor_utils import pad_tensors
from utils.text_utils import fuzzy_find_all, is_number, atof, norm_text, finetune_start
from utils.utils import map_span, find_closest_subseq

logger = logging.getLogger(__name__)


class FairSampler(object):
    """Ensure that all candidates are drawn the same number of times"""

    def __init__(self, candidates):
        self.candidates = list(candidates)
        self.pointer = 0

    def __len__(self):
        return len(self.candidates)

    def sample(self, k: int = None):
        if len(self.candidates) == 0:
            return None if k is None else [None] * k

        if k is None:
            if self.pointer == 0:
                random.shuffle(self.candidates)
            sample_idx = self.pointer
            self.pointer = (self.pointer + 1) % len(self.candidates)
            return self.candidates[sample_idx]

        samples = []
        for _ in range(k):
            if self.pointer == 0:
                random.shuffle(self.candidates)
            sample_idx = self.pointer
            self.pointer = (self.pointer + 1) % len(self.candidates)
            samples.append(self.candidates[sample_idx])
        return samples


class TransitionDataset(Dataset):

    def __init__(self, data_path, tokenizer, corpus, title2id,
                 max_seq_len=512, max_q_len=96, max_obs_len=256,
                 hard_negs_per_state=2, memory_size=3, max_distractors=2, strict=False):
        """

        Args:
            data_path (str):
            tokenizer (ElectraTokenizerFast):
            corpus (dict): passage id -> passage dict
            title2id (dict): norm_title -> passage id
            max_seq_len (int):
            max_q_len (int):
            max_obs_len (int):
            hard_negs_per_state (int):
            memory_size (int):
            max_distractors (int):
            strict (bool):
        """
        self.simple_tokenizer = SimpleTokenizer()
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.title2id = title2id
        self.max_seq_len = max_seq_len
        self.max_q_len = max_q_len
        self.max_obs_len = max_obs_len
        self.hard_negs_per_state = hard_negs_per_state
        self.memory_size = memory_size
        self.max_distractors = max_distractors
        self.strict = strict

        self.q_ids = []
        self.examples = dict()
        self.transitions = []
        with open(data_path) as f:
            for line in f:
                example = json.loads(line)
                q_id = example.pop('_id')
                sp_titles = list(example['sp_facts'].keys())  # unescaped sp titles
                assert len(sp_titles) == 2

                sp_ids = list(title2id[t] for t in sp_titles)
                hn_ids = example['hard_negs']  # all hard negatives
                assert len(hn_ids) == len(set(hn_ids)) > 0 and len(set(sp_ids) & set(hn_ids)) == 0

                transition_offset = len(self.transitions)
                self.transitions.append({"q_id": q_id, "evidences": []})
                for sp_id in sp_ids:
                    self.transitions.append({"q_id": q_id, "evidences": [sp_id]})
                self.transitions.append({"q_id": q_id, "evidences": sp_ids})

                example['transition_offset'] = transition_offset
                example['num_transition'] = len(self.transitions) - transition_offset
                example['transition_sampler'] = FairSampler(list(range(transition_offset, len(self.transitions))))
                example['sp_ids'] = sp_ids
                example['hn_sampler'] = FairSampler(hn_ids)

                # patch and normalize answers
                if q_id == '5a84b9c95542997b5ce3ff35':
                    example['answer'] = 'grandfather'
                elif q_id == '5a8346bb55429966c78a6b69':
                    example['answer'] = 'Lycians'
                elif q_id == '5ae5eb215542996de7b71a6e':
                    example['answer'] = 'Tunisia'
                elif q_id == '5ade61d8554299728e26c703':
                    example['answer'] = "Cecelia Ahern's second novel"
                elif q_id == '5adf9eae5542995ec70e9053':
                    example['answer'] = 'Achaemenid Empire'
                example['answer'] = norm_text(example['answer'])
                if 'answers' in example:
                    example['answers'] = [norm_text(ans) for ans in example['answers']]

                self.q_ids.append(q_id)
                self.examples[q_id] = example

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        q_id = self.q_ids[index]
        example = self.examples[q_id]
        transition = self.transitions[example['transition_sampler'].sample()]

        yes = ADDITIONAL_SPECIAL_TOKENS['YES']
        no = ADDITIONAL_SPECIAL_TOKENS['NO']
        none = ADDITIONAL_SPECIAL_TOKENS['NONE']
        sop = ADDITIONAL_SPECIAL_TOKENS['SOP']
        none_id = self.tokenizer.convert_tokens_to_ids(none)
        sop_id = self.tokenizer.convert_tokens_to_ids(sop)
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        '''
         S                P0       Q        P1                P2
        [CLS] [YES] [NO] [NONE] q [SEP] t1 [SOP] p1 [SEP] t2 [SOP] p2 [SEP]
        '''

        question = f"{yes} {no} {none} {example['question']}"
        answers = example.get('answers', [example['answer']])
        sp_facts = example['sp_facts']  # norm_title - > [sent_idx, ...]
        sp_ids = example['sp_ids']
        state2action = example['state2action']

        # tokenize question
        question_codes = self.tokenizer(question, add_special_tokens=False)
        question_tokens = list(question_codes['input_ids'])

        paras_mark = []
        paras_span = []
        paras_label = []
        sents_span = []
        sents_map = []
        sents_label = []

        dense_expansion_id = None  # no expansion
        dense_expansion = -1  # -1: represent dense query with [SEP] after the question
        link_targets = [none]  # [norm_target, ...]
        links_spans = [[(3, 3)]]  # [((anchor_start, anchor_end), ...), ... ]
        link_label = 0

        # sample some distractors for context
        evidences = transition['evidences']
        n_distractor = random.randint(0, min(self.max_distractors, self.memory_size + 1 - len(evidences)))
        distractors = example['hn_sampler'].sample(k=n_distractor)
        context_ids = np.random.permutation(evidences + distractors).tolist()
        assert len(context_ids) <= self.memory_size + 1

        if len(context_ids) == 0:
            question_tokens_ = question_tokens[:self.max_seq_len - 2]
            token_ids = [cls_id] + question_tokens_ + [sep_id]
            context_token_offset = len(token_ids)
            token_type_ids = [0] * context_token_offset
            answer_mask = [0] * context_token_offset
            for idx in [1, 2, NA_POS]:
                answer_mask[idx] = 1
            assert len(token_ids) == len(token_type_ids) == len(answer_mask) <= self.max_seq_len

            # mark the span of sparse_query
            sparse_query = state2action['initial']['query']
            sparse_query = unescape(sparse_query)
            sparse_query_tokens = self.tokenizer(sparse_query, add_special_tokens=False)['input_ids']
            start_token, end_token, dist = find_closest_subseq(token_ids, sparse_query_tokens,
                                                               max_dist=3, min_ratio=0.75)
            if 0 <= start_token < end_token:  # represent sparse query with its span
                start_token_ = finetune_start(start_token, token_ids, self.tokenizer)
                if start_token != start_token_:
                    logger.debug(f"finetune match 『{self.tokenizer.decode(token_ids[start_token:end_token])}』"
                                 f"->『{self.tokenizer.decode(token_ids[start_token_:end_token])}』")
                    start_token = start_token_
                if dist > 0:
                    logger.debug(f"{q_id}: fuzzy match sparse_query dist={dist} from {len(token_ids)} tokens\n"
                                 f"  origin:  {sparse_query}\n"
                                 f"  matched: {self.tokenizer.decode(token_ids[start_token:end_token])}\n"
                                 f"  from: {self.tokenizer.decode(token_ids)}")
                sparse_start, sparse_end = start_token, end_token - 1
            else:  # represent sparse query with [NONE]
                if len(token_ids) < self.max_seq_len:
                    logger.debug(f"{q_id}: can't find sparse_query 『{sparse_query}』 from {len(token_ids)} tokens")
                    logger.debug(f"Truncated: {self.tokenizer.decode(token_ids)}")
                    logger.debug(sparse_query_tokens)
                    logger.debug(token_ids)
                else:
                    logger.debug(f"{q_id}: can't find sparse_query 『{sparse_query}』 from {len(token_ids)} tokens")
                sparse_start, sparse_end = 3, 3  # len(token_ids) - 1, len(token_ids) - 1
            assert sparse_start <= sparse_end, f"{sparse_start} <= {sparse_end}"

            action = state2action['initial']['action']  # BM25/MDR/ANSWER
            action_label = FUNC2ID[action]

            return {
                "q_id": q_id,
                "context_ids": context_ids,  # (P,)
                "context": "",
                "context_token_spans": [],  # (CT, 2)
                "sents_map": sents_map,
                "sparse_query": sparse_query,
                "dense_expansion_id": dense_expansion_id,  # passage id or None
                "link_targets": link_targets,  # (L,)
                "input_ids": torch.tensor(token_ids, dtype=torch.long),  # (T,)
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),  # (T,)
                "answer_mask": torch.tensor(answer_mask, dtype=torch.float),  # (T,)
                "context_token_offset": torch.tensor(context_token_offset, dtype=torch.long),
                "paras_mark": paras_mark,  # (P,) torch.tensor([], dtype=torch.long)
                "paras_span": paras_span,  # (P, 2) torch.tensor([], dtype=torch.long).reshape(-1, 2)
                "paras_label": torch.tensor(paras_label, dtype=torch.float),  # (P,)
                "sents_span": sents_span,  # (S, 2) torch.tensor([], dtype=torch.long).reshape(-1, 2)
                "sents_label": torch.tensor(sents_label, dtype=torch.float),  # (S,)
                "answer_starts": torch.tensor([NA_POS], dtype=torch.long),  # (A,)
                "answer_ends": torch.tensor([NA_POS], dtype=torch.long),  # (A,)
                "sparse_start": torch.tensor(sparse_start, dtype=torch.long),
                "sparse_end": torch.tensor(sparse_end, dtype=torch.long),
                "dense_expansion": torch.tensor(dense_expansion, dtype=torch.long),
                "links_spans": links_spans,  # (L, _M, 2)
                "link_label": torch.tensor(link_label, dtype=torch.long),
                "action_label": torch.tensor(action_label, dtype=torch.long)
            }

        def updates(p_id):
            nonlocal context
            passage = self.corpus[p_id]
            if self.strict:
                content_start, content_end = passage['sentence_spans'][0][0], passage['sentence_spans'][-1][1]
            else:
                content_start, content_end = 0, len(passage['text'])
            norm_title = unescape(passage['title'])
            content = passage['text'][content_start:content_end]
            # if len(content) == 0:
            #     logger.debug(f"empty text in {passage['title']}({p_id})")
            content_offset = len(context) + len(f"{norm_title} {sop} ")
            # update context
            context += f"{norm_title} {sop} {content}"
            # update spans of sents, labels of sents and paras
            for sent_idx, sent_span in enumerate(passage['sentence_spans']):
                if sent_span[0] >= sent_span[1]:
                    # logger.debug(f"empty {sent_idx}-th sentence {sent_span} in {passage['title']}({p_id})")
                    continue
                sent_span_ = tuple(content_offset + x - content_start for x in sent_span)
                assert context[sent_span_[0]:sent_span_[1]] == passage['text'][sent_span[0]:sent_span[1]]
                sents_char_span.append(sent_span_)
                real_sents_map.append((passage['title'], sent_idx))
                sents_label.append(int(norm_title in sp_facts and sent_idx in sp_facts[norm_title]))
            paras_label.append(int(norm_title in sp_facts))
            # update anchors_spans
            for tgt, mention_spans in get_valid_links(passage, self.strict, self.title2id.get).items():
                if self.title2id[tgt] in context_ids:  # filter hyperlinks
                    continue
                valid_anchor_spans = []
                for anchor_span in mention_spans:
                    if not (0 <= anchor_span[0] - content_start < anchor_span[1] - content_start < len(content)):
                        continue
                    anchor_span_ = tuple(content_offset + x - content_start for x in anchor_span)
                    assert context[anchor_span_[0]:anchor_span_[1]] == passage['text'][anchor_span[0]:anchor_span[1]]
                    valid_anchor_spans.append(anchor_span_)
                if len(valid_anchor_spans) > 0:
                    if tgt not in links_char_spans:
                        links_char_spans[tgt] = valid_anchor_spans
                    else:
                        links_char_spans[tgt].extend(valid_anchor_spans)

        # update context, sentences spans and hyperlinks spans for input, and label sentences and paragraphs
        context = ''
        sents_char_span = []
        real_sents_map = []
        links_char_spans = dict()  # norm_title -> [(s, e), ...]
        for c_idx, para_id in enumerate(context_ids):
            if c_idx != 0:
                context += ' [SEP] '
            updates(para_id)
        assert len(paras_label) == len(context_ids) and len(sents_char_span) == len(real_sents_map) == len(sents_label)

        # tokenize context
        context_codes = self.tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)
        # noinspection PyTypeChecker
        context_tokens, context_token_spans = list(context_codes['input_ids']), list(context_codes['offset_mapping'])
        paras_tokens, para_tokens = [], []
        for token in context_tokens:
            if token == sep_id:
                paras_tokens.append(para_tokens)
                para_tokens = []
            else:
                para_tokens.append(token)
        if len(para_tokens) > 0:
            paras_tokens.append(para_tokens)
        assert len(context_tokens) == len(context_token_spans) == sum(map(len, paras_tokens)) + len(paras_tokens) - 1

        # concatenate question and context
        if 1 + len(question_tokens) + 1 + len(context_tokens) + 1 <= self.max_seq_len:
            question_tokens_ = question_tokens
            context_tokens_ = context_tokens
            context_token_spans_ = context_token_spans
        else:
            question_tokens_ = question_tokens[:max(self.max_q_len, self.max_seq_len - len(context_tokens) - 3)]
            # TODO: truncate each para
            context_tokens_ = context_tokens[:self.max_seq_len - len(question_tokens_) - 3]
            context_token_spans_ = context_token_spans[:len(context_tokens_)]
            assert len(question_tokens_) + len(context_tokens_) + 3 == self.max_seq_len
        token_ids = [cls_id] + question_tokens_ + [sep_id]
        context_token_offset = len(token_ids)
        token_type_ids = [0] * context_token_offset
        answer_mask = [0] * context_token_offset
        for idx in [1, 2, NA_POS]:
            answer_mask[idx] = 1
        token_ids += context_tokens_
        token_type_ids += [1] * len(context_tokens_)
        answer_mask += [int(token_id not in [sop_id, sep_id]) for token_id in context_tokens_]
        if token_ids[-1] != sep_id:
            token_ids.append(sep_id)
            token_type_ids.append(1)
            answer_mask.append(0)
        assert len(token_ids) == len(token_type_ids) == len(answer_mask) <= self.max_seq_len

        # get marks, spans and labels of paragraphs remained in input
        para_start = context_token_offset
        t_idx = para_start + 1
        while t_idx < len(token_ids):
            if token_ids[t_idx] == sep_id:
                assert token_ids[para_start - 1] == sep_id
                paras_span.append((para_start, t_idx - 1))
                para_start = t_idx + 1
            elif token_ids[t_idx] == sop_id:
                paras_mark.append(t_idx)
            t_idx += 1
        paras_label = paras_label[:len(paras_mark)]
        context_ids = context_ids[:len(paras_mark)]
        paras_span = paras_span[:len(paras_mark)]
        last_para_len = len(paras_tokens[len(paras_span) - 1])
        last_para_len_ = paras_span[-1][1] - paras_span[-1][0] + 1
        if last_para_len_ < last_para_len and last_para_len_ < max(0.2 * last_para_len, 12):
            logger.debug(f"remove the last para that is broken and too short ({last_para_len} -> {last_para_len_})")
            # token_ids = token_ids[:paras_span[-1][0]]
            context_ids = context_ids[:-1]
            paras_mark = paras_mark[:-1]
            paras_span = paras_span[:-1]
            paras_label = paras_label[:-1]
        evidences = [p_id for p_id in context_ids if p_id in sp_ids]
        # distractors = [p_id for p_id in context_ids if p_id not in sp_ids]

        # map spans of sentences and hyperlinks for input, and label the link to click
        context_char2token = [-1] * len(context)
        for c_t_idx, (ts, te) in enumerate(context_token_spans_):
            for char_idx in range(ts, te):
                context_char2token[char_idx] = context_token_offset + c_t_idx
        for (start_char, end_char), sent_map in zip(sents_char_span, real_sents_map):
            start_token, end_token = map_span(context_char2token, (start_char, end_char - 1))
            if start_token >= 0:
                sents_span.append((start_token, end_token))
                sents_map.append(sent_map)
        sents_label = sents_label[:len(sents_span)]
        assert len(sents_label) == len(sents_span) == len(sents_map)
        # for target in np.random.permutation(list(links_char_spans.keys())).tolist():
        for target in sorted(links_char_spans.keys(), key=lambda k: (len(links_char_spans[k]), k)):
            char_spans = links_char_spans[target]
            assert self.title2id[target] not in context_ids
            anchor_spans = []  # (_M, 2)
            for start_char, end_char in char_spans:
                start_token, end_token = map_span(context_char2token, (start_char, end_char - 1))
                if start_token >= 0:
                    anchor_spans.append((start_token, end_token))
            if len(anchor_spans) > 0:
                if target in sp_facts.keys():  # label the last link that direct to a unrecalled sp
                    link_label = len(link_targets)
                link_targets.append(target)
                links_spans.append(anchor_spans)
        assert len(link_targets) == len(links_spans) > 0

        # mark the dense query expansion paragraph index in context
        for c_idx, para_id in enumerate(context_ids):
            if para_id in sp_ids:  # represent dense query expansion with the SOP mark of the first SP
                dense_expansion_id = para_id
                dense_expansion = c_idx
                break

        # mark the span of sparse_query
        if len(evidences) == 0:
            sparse_query = state2action['initial']['query']
        elif len(evidences) == 1:
            sparse_query = state2action[unescape(self.corpus[evidences[0]]['title'])]['query']
        else:
            assert len(evidences) == 2
            sparse_query = random.choice([state2action[sp_title]['query'] for sp_title in sp_facts])
        sparse_query = unescape(sparse_query)
        sparse_query = re.sub(r'(^| )<t>( |$)', ' [SEP] ', sparse_query).strip()
        sparse_query = re.sub(r'(^| )</t>( |$)', f' {sop} ', sparse_query).strip()
        sparse_query = max([seg.strip() for seg in sparse_query.split(' [SEP] ')], key=lambda x: len(x.split()))
        sparse_query_tokens = self.tokenizer(sparse_query, add_special_tokens=False)['input_ids']
        start_token, end_token, dist = find_closest_subseq(token_ids, sparse_query_tokens, max_dist=3, min_ratio=0.75)
        if 0 <= start_token < end_token:  # represent sparse query with its span
            start_token_ = finetune_start(start_token, token_ids, self.tokenizer)
            if start_token != start_token_:
                logger.debug(f"finetune match 『{self.tokenizer.decode(token_ids[start_token:end_token])}』"
                             f"->『{self.tokenizer.decode(token_ids[start_token_:end_token])}』")
                start_token = start_token_
            if dist > 0:
                logger.debug(f"{q_id}: fuzzy match sparse_query dist={dist} from {len(token_ids)} tokens\n"
                             f"  origin:  {sparse_query} \n"
                             f"  matched: {self.tokenizer.decode(token_ids[start_token:end_token])}\n"
                             f"  from: {self.tokenizer.decode(token_ids)}")
            sparse_start, sparse_end = start_token, end_token - 1
        else:  # represent sparse query with [NONE]
            if len(token_ids) < self.max_seq_len:
                logger.debug(f"{q_id}: can't find sparse_query 『{sparse_query}』 from {len(token_ids)} tokens")
                logger.debug(f"Original:  [CLS] {question} [SEP] {context} [SEP]")
                logger.debug(f"Truncated: {self.tokenizer.decode(token_ids)}")
                logger.debug(sparse_query_tokens)
                logger.debug(token_ids)
            else:
                logger.debug(f"{q_id}: can't find sparse_query 『{sparse_query}』 from {len(token_ids)} tokens")
            sparse_start, sparse_end = 3, 3  # len(token_ids) - 1, len(token_ids) - 1
        assert sparse_start <= sparse_end, f"{sparse_start} <= {sparse_end}"

        # label spans of answers, don't early answer
        context_ = context[context_token_spans_[0][0]:context_token_spans_[-1][1]]
        if len(evidences) < 2:
            answer_starts, answer_ends = [NA_POS], [NA_POS]
        else:
            assert len(evidences) == 2
            if answers[0].lower() == 'yes':
                answer_starts, answer_ends = [1], [1]
            elif answers[0].lower() == 'no':
                answer_starts, answer_ends = [2], [2]
            else:  # span
                answer_starts, answer_ends = [], []
                char_spans, _ = fuzzy_find_all(context_, answers, self.simple_tokenizer, ignore_case=False)
                for start_char, end_char in char_spans:
                    start_token, end_token = map_span(context_char2token, (start_char, end_char - 1))
                    if start_token >= 0:
                        answer_starts.append(start_token)
                        answer_ends.append(end_token)

                if len(answer_starts) == 0:
                    logger.debug(f"{q_id}: can't find cased ans words in {len(evidences)}/{len(context_ids)} SP "
                                 f"of {len(token_ids)} tokens: {answers}")
                    char_spans, _ = fuzzy_find_all(context_, answers, self.simple_tokenizer, ignore_case=True)
                    for start_char, end_char in char_spans:
                        start_token, end_token = map_span(context_char2token, (start_char, end_char - 1))
                        if start_token >= 0:
                            answer_starts.append(start_token)
                            answer_ends.append(end_token)

                if len(answer_starts) == 0:
                    logger.debug(f"{q_id}: can't find uncased ans words in {len(evidences)}/{len(context_ids)} SP "
                                 f"of {len(token_ids)} tokens: {answers}")
                    char_spans, matches = fuzzy_find_all(context_, answers, self.simple_tokenizer,
                                                         ignore_case=True, max_l_dist=3, min_ratio=0.75)
                    for (start_char, end_char), match in zip(char_spans, matches):
                        if is_number(match) and atof(match) not in [atof(ans) for ans in answers]:
                            continue
                        start_token, end_token = map_span(context_char2token, (start_char, end_char - 1))
                        if start_token >= 0:
                            answer_starts.append(start_token)
                            answer_ends.append(end_token)
                            logger.debug(f"{q_id}: fuzzy match answer {answers} {matches} from {len(token_ids)} tokens")

                if len(answer_starts) == 0:
                    logger.debug(f"{q_id}: can't fuzzy find ans words in {len(evidences)}/{len(context_ids)} SP "
                                 f"of {len(token_ids)} tokens: {answers}")
                    char_spans, matches = fuzzy_find_all(context_, answers, self.simple_tokenizer,
                                                         ignore_case=True, max_l_dist=3, min_ratio=0.75, level='char')
                    for (start_char, end_char), match in zip(char_spans, matches):
                        if is_number(match) and atof(match) not in [atof(ans) for ans in answers]:
                            continue
                        start_token, end_token = map_span(context_char2token, (start_char, end_char - 1))
                        if start_token >= 0:
                            answer_starts.append(start_token)
                            answer_ends.append(end_token)
                            logger.debug(f"{q_id}: fuzzy match answer {answers} {matches} from {len(token_ids)} tokens")

                if len(answer_starts) == 0:  # no answer found in context
                    logger.debug(f"{q_id}: can't fuzzy find ans chars in {len(evidences)}/{len(context_ids)} SP "
                                 f"of {len(token_ids)} tokens: {answers}")
                    answer_starts, answer_ends = [NA_POS], [NA_POS]
        assert all(s <= e for s, e in zip(answer_starts, answer_ends))

        # label action
        if len(evidences) == 0:
            action = 'LINK' if link_label > 0 else state2action['initial']['action']
        elif len(evidences) == 1:
            action = 'LINK' if link_label > 0 else state2action[unescape(self.corpus[evidences[0]]['title'])]['action']
            if action == 'LINK' and link_label == 0:
                sp2_ranks = state2action[unescape(self.corpus[evidences[0]]['title'])]['sp2_ranks']
                best_strategy = min(sp2_ranks.keys(), key=lambda k: sp2_ranks[k])
                alt_action = 'BM25' if best_strategy.startswith('BM25') else 'MDR'
                logger.debug(f"{q_id}: expected link anchor is out of {len(token_ids)} tokens, "
                             f"relabel action with {alt_action}")
                action = alt_action
        else:
            assert len(evidences) == 2 and link_label == 0
            action = 'ANSWER'
        if action == 'BM25' and 3 in (sparse_start, sparse_end):
            logger.debug(f"{q_id}: expected sparse query 『{sparse_query}』 is not found "
                         f"from {len(token_ids)} tokens when {len(evidences)} SP, relabel action with MDR")
            action = 'MDR'
        action_label = FUNC2ID[action]

        # check labels
        if action == 'ANSWER':
            assert len(answer_starts) > 0
            if len(evidences) == 2 and answer_starts == [NA_POS]:
                if len(token_ids) < self.max_seq_len:
                    logger.warning(f"{q_id}: can't find answer: {answers} in {len(evidences)}/{len(context_ids)} SP "
                                   f"of {len(token_ids)} tokens")
                    logger.debug(context_)
                    logger.debug([self.tokenizer(ans, add_special_tokens=False)['input_ids'] for ans in answers])
                    logger.debug(context_tokens_)
                else:
                    logger.debug(f"{q_id}: can't find answer: {answers} in {len(evidences)}/{len(context_ids)} SP "
                                 f"of {len(token_ids)} tokens")
            if len(evidences) < 2 and answer_starts != [NA_POS]:
                logger.warning(f"{q_id}: early answer "
                               f"『{self.tokenizer.decode(token_ids[answer_starts[0]:answer_ends[0] + 1])}』 on SP "
                               f"{evidences}, state2action: {state2action}")
        elif action == 'BM25':
            try:
                assert token_ids[sparse_start:sparse_end + 1] != [none_id]
            except:
                import pdb
                pdb.set_trace()
        elif action == 'MDR':
            if dense_expansion_id in sp_ids:
                assert dense_expansion_id in evidences
            else:
                assert dense_expansion_id is None and len(evidences) == 0
        else:
            assert action == 'LINK' and link_targets[link_label] in sp_facts

        assert (link_label > 0) == (action == 'LINK')
        if action != 'ANSWER':
            assert answer_starts == answer_ends == [NA_POS]
            # ignore (some) negative labels
            answer_starts, answer_ends = [-1], [-1]
            # if random.random() < 0.9:
            #     answer_starts, answer_ends = [-1], [-1]
        if action != 'LINK':
            assert link_label == 0
            # ignore (some) negative labels
            # link_label = -1
            if random.random() < 0.20:
                link_label = -1

        return {
            "q_id": q_id,
            "context_ids": context_ids,  # (P,)
            "context": context,
            "context_token_spans": context_token_spans_,  # (CT, 2)
            "sents_map": sents_map,
            "sparse_query": sparse_query,
            "dense_expansion_id": dense_expansion_id,  # passage id or None
            "link_targets": link_targets,  # (L,)
            "input_ids": torch.tensor(token_ids, dtype=torch.long),  # (T,)
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),  # (T,)
            "answer_mask": torch.tensor(answer_mask, dtype=torch.float),  # (T,)
            "context_token_offset": torch.tensor(context_token_offset, dtype=torch.long),
            "paras_mark": paras_mark,  # (P,) torch.tensor(paras_mark, dtype=torch.long)
            "paras_span": paras_span,  # (P, 2) torch.tensor(paras_span, dtype=torch.long)
            "paras_label": torch.tensor(paras_label, dtype=torch.float),  # (P,)
            "sents_span": sents_span,  # (S, 2) torch.tensor(sents_span, dtype=torch.long).reshape(-1, 2)
            "sents_label": torch.tensor(sents_label, dtype=torch.float),  # (S,)
            "answer_starts": torch.tensor(answer_starts, dtype=torch.long),  # (A,)
            "answer_ends": torch.tensor(answer_ends, dtype=torch.long),  # (A,)
            "sparse_start": torch.tensor(sparse_start, dtype=torch.long),
            "sparse_end": torch.tensor(sparse_end, dtype=torch.long),
            "dense_expansion": torch.tensor(dense_expansion, dtype=torch.long),
            "links_spans": links_spans,  # (L, _M, 2) [torch.LongTensor(link_spans) for link_spans in links_spans]
            "link_label": torch.tensor(link_label, dtype=torch.long),
            "action_label": torch.tensor(action_label, dtype=torch.long)
        }


class ActionDataset(Dataset):

    def __init__(self, samples: List[dict], tokenizer: ElectraTokenizerFast, title2id: Callable,
                 max_seq_len: int = 512, max_q_len: int = 96, max_obs_len: int = 256, strict: bool = False):
        self.samples = samples
        self.simple_tokenizer = SimpleTokenizer()
        self.tokenizer = tokenizer
        self.title2id = title2id
        self.max_seq_len = max_seq_len
        self.max_q_len = max_q_len
        self.max_obs_len = max_obs_len
        self.strict = strict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        q_id = sample['q_id']

        yes = ADDITIONAL_SPECIAL_TOKENS['YES']
        no = ADDITIONAL_SPECIAL_TOKENS['NO']
        none = ADDITIONAL_SPECIAL_TOKENS['NONE']
        sop = ADDITIONAL_SPECIAL_TOKENS['SOP']
        sop_id = self.tokenizer.convert_tokens_to_ids(sop)
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        '''
         S                P0       Q        P1                P2
        [CLS] [YES] [NO] [NONE] q [SEP] t1 [SOP] p1 [SEP] t2 [SOP] p2 [SEP]
        '''

        question = f"{yes} {no} {none} {sample['question']}"

        # tokenize question
        question_codes = self.tokenizer(question, add_special_tokens=False)
        # noinspection PyTypeChecker
        question_tokens = list(question_codes['input_ids'])

        paras_mark = []
        paras_span = []
        sents_span = []
        sents_map = []

        link_targets = [none]  # [norm_target, ...]
        links_spans = [[(3, 3)]]  # [((anchor_start, anchor_end), ...), ... ]

        context_ids = sample['context_ids']
        passages = sample['passages']
        sparse_query = sample['sparse_query']
        sparse_query = unescape(sparse_query)
        excluded = sample['excluded']

        if context_ids[0] is None:
            question_tokens_ = question_tokens[:self.max_seq_len - 2]
            token_ids = [cls_id] + question_tokens_ + [sep_id]
            context_token_offset = len(token_ids)
            token_type_ids = [0] * context_token_offset
            answer_mask = [0] * context_token_offset
            for idx in [1, 2, NA_POS]:
                answer_mask[idx] = 1
            assert len(token_ids) == len(token_type_ids) == len(answer_mask) <= self.max_seq_len

            # mark the span of sparse_query
            sparse_query_tokens = self.tokenizer(sparse_query, add_special_tokens=False)['input_ids']
            start_token, end_token, dist = find_closest_subseq(token_ids, sparse_query_tokens,
                                                               max_dist=3, min_ratio=0.75)
            if 0 <= start_token < end_token:  # represent sparse query with its span
                start_token_ = finetune_start(start_token, token_ids, self.tokenizer)
                if start_token != start_token_:
                    logger.debug(f"finetune match 『{self.tokenizer.decode(token_ids[start_token:end_token])}』"
                                 f"->『{self.tokenizer.decode(token_ids[start_token_:end_token])}』")
                    start_token = start_token_
                if dist > 0:
                    logger.debug(f"{q_id}: fuzzy match sparse_query dist={dist} from {len(token_ids)} tokens\n"
                                 f"  origin:  {sparse_query}\n"
                                 f"  matched: {self.tokenizer.decode(token_ids[start_token:end_token])}\n"
                                 f"  from: {self.tokenizer.decode(token_ids)}")
                sparse_start, sparse_end = start_token, end_token - 1
            else:  # represent sparse query with [NONE]
                if len(token_ids) < self.max_seq_len:
                    logger.debug(f"{q_id}: can't find sparse_query 『{sparse_query}』 from {len(token_ids)} tokens")
                    logger.debug(f"Truncated: {self.tokenizer.decode(token_ids)}")
                    logger.debug(sparse_query_tokens)
                    logger.debug(token_ids)
                else:
                    logger.debug(f"{q_id}: can't find sparse_query 『{sparse_query}』 from {len(token_ids)} tokens")
                sparse_start, sparse_end = 3, 3  # len(token_ids) - 1, len(token_ids) - 1
            assert sparse_start <= sparse_end, f"{sparse_start} <= {sparse_end}"

            return {
                "q_id": q_id,
                "question": sample['question'],
                "context_ids": [],  # (P,)
                "context": "",
                "context_token_spans": [],  # (CT, 2)
                "sents_map": sents_map,
                "sparse_query": sparse_query,  # sample['sparse_query'],
                "link_targets": link_targets,  # (L,)
                "input_ids": torch.tensor(token_ids, dtype=torch.long),  # (T,)
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),  # (T,)
                "answer_mask": torch.tensor(answer_mask, dtype=torch.float),  # (T,)
                "context_token_offset": torch.tensor(context_token_offset, dtype=torch.long),
                "paras_mark": paras_mark,  # (P,) torch.tensor([], dtype=torch.long)
                "paras_span": paras_span,  # (P, 2) torch.tensor([], dtype=torch.long).reshape(-1, 2)
                "sents_span": sents_span,  # (S, 2) torch.tensor([], dtype=torch.long).reshape(-1, 2)
                "sparse_start": torch.tensor(sparse_start, dtype=torch.long),
                "sparse_end": torch.tensor(sparse_end, dtype=torch.long),
                "links_spans": links_spans  # (L, _M, 2)
            }

        def updates(para):
            nonlocal context
            if self.strict:
                content_start, content_end = para['sentence_spans'][0][0], para['sentence_spans'][-1][1]
            else:
                content_start, content_end = 0, len(para['text'])
            content = para['text'][content_start:content_end]
            # if len(content) == 0:
            #     logger.debug(f"empty text in {para['title']}")
            content_offset = len(context) + len(f"{unescape(para['title'])} {sop} ")
            # update context
            context += f"{unescape(para['title'])} {sop} {content}"
            # update spans of sents
            for sent_idx, sent_span in enumerate(para['sentence_spans']):
                if sent_span[0] >= sent_span[1]:
                    # logger.debug(f"empty {sent_idx}-th sentence {sent_span} in {para['title']}")
                    continue
                sent_span_ = tuple(content_offset + x - content_start for x in sent_span)
                assert context[sent_span_[0]:sent_span_[1]] == para['text'][sent_span[0]:sent_span[1]]
                sents_char_span.append(sent_span_)
                real_sents_map.append((para['title'], sent_idx))
            # update anchors_spans
            for tgt, mention_spans in get_valid_links(para, self.strict, self.title2id).items():
                if self.title2id(tgt) in context_ids or self.title2id(tgt) in excluded:  # filter hyperlinks
                    continue
                valid_anchor_spans = []
                for anchor_span in mention_spans:
                    if not (0 <= anchor_span[0] - content_start < anchor_span[1] - content_start < len(content)):
                        continue
                    anchor_span_ = tuple(content_offset + x - content_start for x in anchor_span)
                    assert context[anchor_span_[0]:anchor_span_[1]] == para['text'][anchor_span[0]:anchor_span[1]]
                    valid_anchor_spans.append(anchor_span_)
                if len(valid_anchor_spans) > 0:
                    if tgt not in links_char_spans:
                        links_char_spans[tgt] = valid_anchor_spans
                    else:
                        links_char_spans[tgt].extend(valid_anchor_spans)

        # update context, sentences spans and hyperlinks spans for input, and label sentences and paragraphs
        context = ''
        sents_char_span = []
        real_sents_map = []
        links_char_spans = dict()  # norm_title -> [(s, e), ...]
        for c_idx, passage in enumerate(passages):
            if c_idx != 0:
                context += ' [SEP] '
            updates(passage)
        assert len(sents_char_span) == len(real_sents_map)

        # tokenize context
        context_codes = self.tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)
        # noinspection PyTypeChecker
        context_tokens, context_token_spans = list(context_codes['input_ids']), list(context_codes['offset_mapping'])
        paras_tokens, para_tokens = [], []
        for token in context_tokens:
            if token == sep_id:
                paras_tokens.append(para_tokens)
                para_tokens = []
            else:
                para_tokens.append(token)
        if len(para_tokens) > 0:
            paras_tokens.append(para_tokens)
        assert len(context_tokens) == len(context_token_spans) == sum(map(len, paras_tokens)) + len(paras_tokens) - 1

        # concatenate question and context
        if 1 + len(question_tokens) + 1 + len(context_tokens) + 1 <= self.max_seq_len:
            question_tokens_ = question_tokens
            context_tokens_ = context_tokens
            context_token_spans_ = context_token_spans
        else:
            logger.debug(f"{q_id}: too many tokens({len(question_tokens)}+{len(context_tokens)}+3) "
                         f"of {len(passages)} passages")
            question_tokens_ = question_tokens[:max(self.max_q_len, self.max_seq_len - len(context_tokens) - 3)]
            # TODO: truncate each para
            context_tokens_ = context_tokens[:self.max_seq_len - len(question_tokens_) - 3]
            context_token_spans_ = context_token_spans[:len(context_tokens_)]
            assert len(question_tokens_) + len(context_tokens_) + 3 == self.max_seq_len
        token_ids = [cls_id] + question_tokens_ + [sep_id]
        context_token_offset = len(token_ids)
        token_type_ids = [0] * context_token_offset
        answer_mask = [0] * context_token_offset
        for idx in [1, 2, NA_POS]:
            answer_mask[idx] = 1
        token_ids += context_tokens_
        token_type_ids += [1] * len(context_tokens_)
        answer_mask += [int(token_id not in [sop_id, sep_id]) for token_id in context_tokens_]
        if token_ids[-1] != sep_id:
            token_ids.append(sep_id)
            token_type_ids.append(1)
            answer_mask.append(0)
        assert len(token_ids) == len(token_type_ids) == len(answer_mask) <= self.max_seq_len

        # get marks, spans of paragraphs remained in input
        para_start = context_token_offset
        t_idx = para_start + 1
        while t_idx < len(token_ids):
            if token_ids[t_idx] == sep_id:
                assert token_ids[para_start - 1] == sep_id
                paras_span.append((para_start, t_idx - 1))
                para_start = t_idx + 1
            elif token_ids[t_idx] == sop_id:
                paras_mark.append(t_idx)
            t_idx += 1
        context_ids = context_ids[:len(paras_mark)]
        paras_span = paras_span[:len(paras_mark)]
        # last_para_len = len(paras_tokens[len(paras_span) - 1])
        # last_para_len_ = paras_span[-1][1] - paras_span[-1][0] + 1
        # if last_para_len_ < last_para_len and last_para_len_ < max(0.2 * last_para_len, 12):
        #     logger.debug(f"remove the last para that is broken and too short ({last_para_len} -> {last_para_len_})")
        #     token_ids = token_ids[:paras_span[-1][0]]
        #     context_ids = context_ids[:-1]
        #     paras_mark = paras_mark[:-1]
        #     paras_span = paras_span[:-1]

        # map spans of sentences and hyperlinks for input
        context_char2token = [-1] * len(context)
        for c_t_idx, (ts, te) in enumerate(context_token_spans_):
            for char_idx in range(ts, te):
                context_char2token[char_idx] = context_token_offset + c_t_idx
        for (start_char, end_char), sent_map in zip(sents_char_span, real_sents_map):
            start_token, end_token = map_span(context_char2token, (start_char, end_char - 1))
            if start_token >= 0:
                sents_span.append((start_token, end_token))
                sents_map.append(sent_map)
        assert len(sents_span) == len(sents_map)
        for target in links_char_spans.keys():
            char_spans = links_char_spans[target]
            assert self.title2id(target) not in context_ids and self.title2id(target) not in excluded
            anchor_spans = []  # (_M, 2)
            for start_char, end_char in char_spans:
                start_token, end_token = map_span(context_char2token, (start_char, end_char - 1))
                if start_token >= 0:
                    anchor_spans.append((start_token, end_token))
            if len(anchor_spans) > 0:
                link_targets.append(target)
                links_spans.append(anchor_spans)
        assert len(link_targets) == len(links_spans)

        # mark the span of sparse_query
        sparse_query = re.sub(r'(^| )<t>( |$)', ' [SEP] ', sparse_query).strip()
        sparse_query = re.sub(r'(^| )</t>( |$)', f' {sop} ', sparse_query).strip()
        sparse_query = max([seg.strip() for seg in sparse_query.split(' [SEP] ')], key=lambda x: len(x.split()))
        sparse_query_tokens = self.tokenizer(sparse_query, add_special_tokens=False)['input_ids']
        start_token, end_token, dist = find_closest_subseq(token_ids, sparse_query_tokens, max_dist=3, min_ratio=0.75)
        if 0 <= start_token < end_token:  # represent sparse query with its span
            start_token_ = finetune_start(start_token, token_ids, self.tokenizer)
            if start_token != start_token_:
                logger.debug(f"finetune match 『{self.tokenizer.decode(token_ids[start_token:end_token])}』"
                             f"->『{self.tokenizer.decode(token_ids[start_token_:end_token])}』")
                start_token = start_token_
            if dist > 0:
                logger.debug(f"{q_id}: fuzzy match sparse_query dist={dist} from {len(token_ids)} tokens\n"
                             f"  origin:  {sparse_query} \n"
                             f"  matched: {self.tokenizer.decode(token_ids[start_token:end_token])}\n"
                             f"  from: {self.tokenizer.decode(token_ids)}")
            sparse_start, sparse_end = start_token, end_token - 1
        else:  # represent sparse query with [NONE]
            if len(token_ids) < self.max_seq_len:
                logger.debug(f"{q_id}: can't find sparse_query 『{sparse_query}』 from {len(token_ids)} tokens")
                logger.debug(f"Original:  [CLS] {question} [SEP] {context} [SEP]")
                logger.debug(f"Truncated: {self.tokenizer.decode(token_ids)}")
                logger.debug(sparse_query_tokens)
                logger.debug(token_ids)
            else:
                logger.debug(f"{q_id}: can't find sparse_query 『{sparse_query}』 from {len(token_ids)} tokens")
            sparse_start, sparse_end = 3, 3  # len(token_ids) - 1, len(token_ids) - 1
        assert sparse_start <= sparse_end, f"{sparse_start} <= {sparse_end}"

        return {
            "q_id": q_id,
            "question": sample['question'],
            "context_ids": context_ids,  # (P,)
            "context": context,
            "context_token_spans": context_token_spans_,  # (CT, 2)
            "sents_map": sents_map,  # (S, 2)
            "sparse_query": sparse_query,  # sample['sparse_query'],
            "link_targets": link_targets,  # (L,)
            "input_ids": torch.tensor(token_ids, dtype=torch.long),  # (T,)
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),  # (T,)
            "answer_mask": torch.tensor(answer_mask, dtype=torch.float),  # (T,)
            "context_token_offset": torch.tensor(context_token_offset, dtype=torch.long),
            "paras_mark": paras_mark,  # (P,) torch.tensor(paras_mark, dtype=torch.long)
            "paras_span": paras_span,  # (P, 2) torch.tensor(paras_span, dtype=torch.long).reshape(-1, 2)
            "sents_span": sents_span,  # (S, 2) torch.tensor(sents_span, dtype=torch.long).reshape(-1, 2)
            "sparse_start": torch.tensor(sparse_start, dtype=torch.long),
            "sparse_end": torch.tensor(sparse_end, dtype=torch.long),
            "links_spans": links_spans  # (L, _M, 2) [torch.LongTensor(link_spans) for link_spans in links_spans]
        }


class ConstantDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_transitions(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    nn_input = {
        "input_ids": pad_tensors([sample['input_ids'] for sample in samples], pad_id),  # (B, T)
        "attention_mask": pad_tensors([torch.ones_like(sample['input_ids']) for sample in samples], 0),  # (B, T)
        "token_type_ids": pad_tensors([sample['token_type_ids'] for sample in samples], 0),  # (B, T)
        "answer_mask": pad_tensors([sample['answer_mask'] for sample in samples], 0),  # (B, T)
        "context_token_offset": torch.stack([sample['context_token_offset'] for sample in samples]),  # (B,)
        "paras_mark": [sample['paras_mark'] for sample in samples],  # (B, _P)
        "paras_span": [sample['paras_span'] for sample in samples],  # (B, _P, 2)
        # "paras_label": [sample['paras_label'] for sample in samples],  # (B, _P)
        "sents_span": [sample['sents_span'] for sample in samples],  # (B, _S, 2)
        # "sents_label": [sample['sents_label'] for sample in samples],  # (B, _S)
        # "answer_starts": pad_tensors([sample['answer_starts'] for sample in samples], -1),  # (B, A)
        # "answer_ends": pad_tensors([sample['answer_ends'] for sample in samples], -1),  # (B, A)
        "sparse_start": torch.stack([sample['sparse_start'] for sample in samples]),  # (B,)
        "sparse_end": torch.stack([sample['sparse_end'] for sample in samples]),  # (B,)
        # "dense_expansion": torch.stack([sample['dense_expansion'] for sample in samples]),  # (B,)
        "links_spans": [sample['links_spans'] for sample in samples],  # (B, _L, _M, 2)
        # "link_label": torch.stack([sample['link_label'] for sample in samples]),  # (B,)
        # "action_label": torch.stack([sample['action_label'] for sample in samples]),  # (B,)
    }

    if 'action_label' in samples[0]:
        nn_input['paras_label'] = [sample['paras_label'] for sample in samples]  # (B, _P)
        nn_input['sents_label'] = [sample['sents_label'] for sample in samples]  # (B, _S)
        nn_input['answer_starts'] = pad_tensors([sample['answer_starts'] for sample in samples], -1)  # (B, A)
        nn_input['answer_ends'] = pad_tensors([sample['answer_ends'] for sample in samples], -1)  # (B, A)
        nn_input['dense_expansion'] = torch.stack([sample['dense_expansion'] for sample in samples])  # (B,)
        nn_input['link_label'] = torch.stack([sample['link_label'] for sample in samples])  # (B,)
        nn_input['action_label'] = torch.stack([sample['action_label'] for sample in samples])  # (B,)

    batch = {key: [] for key in samples[0] if key not in nn_input}
    for sample in samples:
        for k in batch:
            batch[k].append(sample[k])
    batch['nn_input'] = nn_input

    return batch
