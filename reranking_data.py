import json
import logging

import torch
from torch.utils.data import Dataset

from transformers import ElectraTokenizerFast  # , PreTrainedTokenizer, PreTrainedTokenizerFast

from basic_tokenizer import SimpleTokenizer

from config import ADDITIONAL_SPECIAL_TOKENS
from utils.tensor_utils import pad_tensors

logger = logging.getLogger(__name__)


class RerankingDataset(Dataset):

    def __init__(self, data_path, tokenizer, corpus, title2id,
                 max_seq_len=512, max_q_len=96, max_obs_len=256, strict=False):
        """

        Args:
            data_path (str):
            tokenizer (ElectraTokenizerFast):
            corpus (dict): passage id -> passage dict
            title2id (dict):
            max_seq_len (int):
            max_q_len (int):
            max_obs_len (int):
            strict (bool):
        """
        self.simple_tokenizer = SimpleTokenizer()
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.title2id = title2id
        self.max_seq_len = max_seq_len
        self.max_q_len = max_q_len
        self.max_obs_len = max_obs_len
        self.strict = strict

        self.qid2idx = dict()
        self.examples = []
        self.qp_pairs = []
        with open(data_path) as f:
            for line in f:
                example = json.loads(line)
                q_id = example['_id']
                sp_titles = list(example['sp_facts'].keys())
                sp_ids = list(title2id[t] for t in sp_titles)
                hn_ids = list(set(example['hard_negs']) - set(sp_ids))
                assert len(sp_ids) == 2 and len(hn_ids) > 0

                pair_offset = len(self.qp_pairs)
                for p_id in sp_ids:
                    self.qp_pairs.append((q_id, [p_id]))
                for p_id in hn_ids:
                    self.qp_pairs.append((q_id, [p_id]))

                example['sp_ids'] = sp_ids
                example['hn_ids'] = hn_ids
                example['pair_offset'] = pair_offset
                example['num_pair'] = len(self.qp_pairs) - pair_offset

                self.qid2idx[q_id] = len(self.examples)
                self.examples.append(example)

    def __len__(self):
        return len(self.qp_pairs)

    def __getitem__(self, index):
        q_id, context_ids = self.qp_pairs[index]
        example = self.examples[self.qid2idx[q_id]]

        yes = ADDITIONAL_SPECIAL_TOKENS['YES']
        no = ADDITIONAL_SPECIAL_TOKENS['NO']
        none = ADDITIONAL_SPECIAL_TOKENS['NONE']
        sop = ADDITIONAL_SPECIAL_TOKENS['SOP']
        sop_id = self.tokenizer.convert_tokens_to_ids(sop)
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        '''
         S                         DQ                                  SQ
        [CLS] [YES] [NO] [NONE] q [SEP] t1 [SOP] p1 [SEP] t2 [SOP] p2 [SEP]
        '''

        question = f"{yes} {no} {none} {example['question']}"

        # tokenize question
        question_codes = self.tokenizer(question, add_special_tokens=False)
        # noinspection PyTypeChecker
        question_tokens = list(question_codes['input_ids'])

        context = ''
        for c_idx, para_id in enumerate(context_ids):
            if c_idx != 0:
                context += ' [SEP] '
            para = self.corpus[para_id]
            if self.strict:
                content_start, content_end = para['sentence_spans'][0][0], para['sentence_spans'][-1][1]
            else:
                content_start, content_end = 0, len(para['text'])
            content = para['text'][content_start:content_end].strip()
            if len(content) == 0:
                logger.debug(f"empty text in {para['title']}")
            context += f"{para['title']} {sop} {content}"

        # tokenize context
        context_codes = self.tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)
        # noinspection PyTypeChecker
        context_tokens, context_token_spans = list(context_codes['input_ids']), list(context_codes['offset_mapping'])
        assert len(context_tokens) == len(context_token_spans)

        # concatenate question and context
        if 1 + len(question_tokens) + 1 + len(context_tokens) + 1 <= self.max_seq_len:
            question_tokens_ = question_tokens
            context_tokens_ = context_tokens
        else:
            logger.debug(f"{q_id}: too many tokens({len(question_tokens)}+{len(context_tokens)}+3) "
                         f"of {len(context_ids)} passages")
            question_tokens_ = question_tokens[:self.max_q_len]
            context_tokens_ = context_tokens[:self.max_seq_len - len(question_tokens_) - 3]
            if len(question_tokens_) + len(context_tokens_) + 3 < self.max_seq_len:
                question_tokens_ = question_tokens[:self.max_seq_len - len(context_tokens_) - 3]
        token_ids = [cls_id] + question_tokens_ + [sep_id]
        context_token_offset = len(token_ids)
        token_type_ids = [0] * context_token_offset
        token_ids += context_tokens_ + [sep_id]
        token_type_ids += [1] * (len(context_tokens_) + 1)
        assert len(token_ids) == len(token_type_ids) <= self.max_seq_len

        # get marks of paragraphs remained in input
        paras_mark = []
        for t_idx, t_id in enumerate(token_ids):
            if t_id == sop_id:
                paras_mark.append(t_idx)
        context_ids = context_ids[:len(paras_mark)]

        return {
            "q_id": q_id,
            "context_ids": context_ids,  # (P,)
            "paras_mark": paras_mark,  # (P,)
            "input_ids": torch.tensor(token_ids, dtype=torch.long),  # (T,)
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),  # (T,)
        }


def collate_qp(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    nn_input = {
        "input_ids": pad_tensors([sample['input_ids'] for sample in samples], pad_id),  # (B, T)
        "attention_mask": pad_tensors([torch.ones_like(sample['input_ids']) for sample in samples], pad_id),  # (B, T)
        "token_type_ids": pad_tensors([sample['token_type_ids'] for sample in samples], 0),  # (B, T)
        "paras_mark": [sample['paras_mark'] for sample in samples],  # (B, _P)
    }

    if 'paras_label' in samples[0]:
        nn_input['paras_label'] = [sample['paras_label'] for sample in samples]  # (B, _P)

    batch = {key: [] for key in samples[0] if key not in nn_input}
    for sample in samples:
        for k in batch:
            batch[k].append(sample[k])
    batch['nn_input'] = nn_input

    return batch
