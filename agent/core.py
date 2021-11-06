from collections import defaultdict
import copy
from functools import partial
from html import unescape
import logging
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from redis import Redis
from transformers import ElectraTokenizerFast

from drqa.reader import Predictor

from config import FUNCTIONS, NA_POS
from hotpot_evaluate_plus import f1_score, normalize_answer
from models.union_model import UnionModel
from transition_data import ActionDataset, collate_transitions
from utils.tensor_utils import to_device
from utils.text_utils import predict_answer
from env.core import BaseEnv

logger = logging.getLogger(__name__)


class Agent(object):

    def __init__(self, tokenizer, union_model, query_generator1, query_generator2, device, env, query_redis,
                 func_mask=(1, 1, 1, 1), memory_size=2, max_seq_len=512, max_q_len=96, max_obs_len=256, strict=False,
                 gold_qas_map=None, oracle_belief=False, oracle_state2action=None):
        """

        Args:
            tokenizer (ElectraTokenizerFast):
            union_model (UnionModel):
            query_generator1 (Predictor):
            query_generator2 (Predictor):
            device (torch.device):
            env (BaseEnv):
            query_redis (Redis):
            func_mask (tuple):
            memory_size (int):
            max_seq_len (int):
            max_q_len (int):
            max_obs_len (int):
            strict (bool): only use the content of introductory paragraph in HotpotQA
        """
        self.tokenizer = tokenizer
        self.union_model = union_model
        self.query_generator1 = query_generator1
        self.query_generator2 = query_generator2
        self.device = device

        self.mode = 'eval'
        self.union_model.eval()

        self.env = env
        self.query_redis = query_redis

        self.masked_functions = [func for func, mask in zip(FUNCTIONS, func_mask) if mask == 0]

        self.memory_size = memory_size
        self.max_seq_len = max_seq_len
        self.max_q_len = max_q_len
        self.max_obs_len = max_obs_len
        self.strict = strict
        self.batch_size = 32

        self.threshold = 0.5
        self.sp_threshold = 0.5

        self.observed = defaultdict(list)  # game_id -> p_ids
        self.obs_scores = defaultdict(list)  # game_id -> p_ids
        self.clicked = defaultdict(list)  # game_id -> p_ids
        self.memory = defaultdict(dict)  # game_id -> {p_id -> score}
        self.discarded = defaultdict(list)  # game_id -> p_ids

        self.proposals = defaultdict(list)
        self.commands = defaultdict(list)
        self.exhausted_cmds = defaultdict(set)  # game_id -> cmds

        self.answer = dict()  # game_id -> ans
        self.all_answer = defaultdict(dict)  # game_id -> {ans -> cumulative prob}
        self.sp_facts = defaultdict(dict)  # game_id -> {(t, s_idx) -> prob}
        self.all_sp_facts = defaultdict(dict)  # game_id -> {(t, s_idx) -> prob}
        self.sp_passages = dict()  # game_id -> titles

        self.cases = defaultdict(set)  # type -> game_ids

        self.gold_qas_map = gold_qas_map
        self.oracle_belief = oracle_belief
        self.oracle_state2action = oracle_state2action
        if self.oracle_state2action:
            self.oracle_belief = True

    def reset(self):
        self.observed.clear()
        self.obs_scores.clear()
        self.clicked.clear()
        self.memory.clear()
        self.discarded.clear()

        self.proposals.clear()
        self.commands.clear()
        self.exhausted_cmds.clear()

        self.answer.clear()
        self.all_answer.clear()
        self.sp_facts.clear()
        self.all_sp_facts.clear()
        self.sp_passages.clear()

        self.cases.clear()

    def train(self):
        """Tell the agent that it's training phase."""
        self.mode = "train"
        self.union_model.train()

    def eval(self):
        """Tell the agent that it's evaluation phase."""
        self.mode = "eval"
        self.union_model.eval()

    def ids2titles(self, game_ids):
        return [self.env.get(g_id)['title'] for g_id in game_ids]

    def pretty_memory(self, game_id):
        return {self.env.get(k)['title']: v for k, v in self.memory[game_id].items()}

    def pretty_cmd(self, command):
        if command[0] == 'MDR':
            if command[1][1] is not None:
                return f"MDR(Q + {repr(self.env.get(command[1][1])['title'])})"
            return "MDR(Q)"
        return f"{command[0]}({repr(command[1])})"

    def pretty_behavior(self, game_id):
        pretty_cmds = []
        last_cmd = None
        size = 1
        for cmd in self.commands[game_id]:
            if self.pretty_cmd(cmd) == last_cmd:
                size += 1
            else:
                if last_cmd is not None:
                    if size > 1:
                        pretty_cmds.append(last_cmd[:-1] + f', {size})')
                    else:
                        pretty_cmds.append(last_cmd)
                last_cmd = self.pretty_cmd(cmd)
                size = 1
        if last_cmd is not None:
            if size > 1:
                pretty_cmds.append(last_cmd[:-1] + f', {size})')
            else:
                pretty_cmds.append(last_cmd)

        return ' / '.join(pretty_cmds)

    def gen_sparse_query(self, game_ids, questions, observations, disable_tqdm=False):
        """generate and cache sparse queries

        Args:
            game_ids (list[str]):
            questions (list[str]):
            observations (list[str]):
            disable_tqdm (bool):

        Returns:
            list[str]:
        """
        sparse_queries = []
        for g_id, question, obs_id in tqdm(zip(game_ids, questions, observations), desc='gen_sparse_query',
                                           total=len(game_ids), disable=disable_tqdm):
            qg_paras = [question]
            obs = self.env.get(obs_id)
            if obs is not None:
                if self.strict:
                    content_start, content_end = obs['sentence_spans'][0][0], obs['sentence_spans'][-1][1]
                else:
                    content_start, content_end = 0, len(obs['text'])
                qg_paras.append(f"<t> {obs['title']} </t> {obs['text'][content_start:content_end]}")
            for p_id in self.memory[g_id].keys():  # xxx sorted(, key=lambda x: -self.memory[g_id][x]):
                if p_id == obs_id:
                    continue
                if len(qg_paras) > 5:
                    break
                para = self.env.get(p_id)
                if self.strict:
                    content_start, content_end = para['sentence_spans'][0][0], para['sentence_spans'][-1][1]
                else:
                    content_start, content_end = 0, len(para['text'])
                qg_paras.append(f"<t> {para['title']} </t> {para['text'][content_start:content_end]}")
            qg_context = ' '.join(qg_paras)

            if self.query_redis.exists(qg_context):
                sparse_queries.append(self.query_redis.get(qg_context))
            else:
                if len(qg_paras) == 1:
                    sparse_query = self.query_generator1.predict(qg_context, question)[0][0].strip()
                else:
                    sparse_query = self.query_generator2.predict(qg_context, question)[0][0].strip()
                self.query_redis.set(qg_context, sparse_query)
                sparse_queries.append(sparse_query)

        return sparse_queries

    def act(self, game_ids, questions, observations=None, review=False, disable_tqdm=False):
        if observations is None:
            observations = [None] * len(game_ids)
        review &= not self.oracle_belief
        disable_tqdm |= len(game_ids) <= 10 * self.batch_size

        sparse_queries = self.gen_sparse_query(game_ids, questions, observations, disable_tqdm=disable_tqdm)

        samples = []
        for g_id, question, obs_id, sparse_query in zip(game_ids, questions, observations, sparse_queries):
            context_ids = sorted(self.memory[g_id].keys(), key=lambda x: -self.memory[g_id][x])  # [:3]
            if obs_id is not None:
                self.observed[g_id].append(obs_id)
                if len(self.commands[g_id]) > 0 and self.commands[g_id][-1][0] == 'LINK':
                    self.clicked[g_id].append(obs_id)
                while obs_id in context_ids:
                    context_ids.remove(obs_id)
                context_ids = [obs_id] + context_ids
            elif len(self.commands[g_id]) > 0:
                logger.debug(f"{g_id}: observed none by {self.pretty_cmd(self.commands[g_id][-1])} "
                             f"when memory={self.memory[g_id]}")
                self.exhausted_cmds[g_id].add(self.commands[g_id][-1])
            if review:
                for (p_title, s_idx), s_prob in sorted(self.all_sp_facts[g_id].items(), key=lambda x: -x[1]):
                    if len(context_ids) >= 3:  # xxx
                        break
                    p_id = self.env.title2id(unescape(p_title))
                    if p_id not in context_ids:
                        context_ids.append(p_id)
                for p_id, p_prob in sorted(zip(self.observed[g_id], self.obs_scores[g_id]), key=lambda x: -x[1]):
                    if len(context_ids) >= 3:  # xxx
                        break
                    if p_id not in context_ids:
                        context_ids.append(p_id)
            if len(context_ids) == 0:
                context_ids.append(None)
            samples.append({
                "q_id": g_id,
                "question": question,
                "context_ids": context_ids,
                "passages": [self.env.get(p_id) for p_id in context_ids],
                "sparse_query": sparse_query,
                "excluded": set(self.clicked[g_id])
            })

        dataset = ActionDataset(samples, self.tokenizer, self.env.title2id,
                                max_seq_len=self.max_seq_len, max_q_len=self.max_q_len, max_obs_len=self.max_obs_len,
                                strict=self.strict)
        collate_func = partial(collate_transitions, pad_id=self.tokenizer.pad_token_id)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_func,
                                pin_memory=True, num_workers=0)

        self.union_model.eval()
        next_commands = {}
        for batch in tqdm(dataloader, desc='act', total=len(dataloader), disable=disable_tqdm):
            nn_input = to_device(batch['nn_input'], self.device)
            with torch.no_grad():
                if self.oracle_belief:
                    oracle_para_logits = [
                        torch.tensor([20. * float(self.env.get(p_id)['title'] in self.gold_qas_map[g_id][2]) - 10.
                                      for p_id in p_ids], device=self.device)
                        for p_ids in batch['context_ids']
                    ]
                    outputs = self.union_model(nn_input, oracle_para_logits)
                else:
                    outputs = self.union_model(nn_input)

                action_logits = outputs[0]  # (B, 4)
                # (B, _P)    (B, _S)      (B,)            (B,)
                para_logits, sent_logits, para_threshold, sent_threshold = outputs[4:8]
                # (B,)       (B,)        (B,)      (B,)       (B,)
                pred_action, pred_start, pred_end, pred_link, pred_exp = outputs[8:13]
                # (B,)    (B,)       (B,)
                ans_conf, link_conf, exp_conf = outputs[13:]

                action_probs = action_logits.softmax(-1)
                para_probs = [_para_logits.sigmoid() for _para_logits in para_logits]  # (B, _P)
                sent_probs = [_sent_logits.sigmoid() for _sent_logits in sent_logits]  # (B, _S)
                para_threshold = para_threshold.sigmoid()  # (B,)
                sent_threshold = sent_threshold.sigmoid()  # (B,)

                for i, g_id in enumerate(batch['q_id']):
                    if self.gold_qas_map is None:
                        gold_ans, gold_sp = None, None
                    else:
                        gold_ans, gold_sp = self.gold_qas_map.get(g_id, (None, None, None))[1:]

                    # update memory
                    miss_sp = None
                    assert len(batch['context_ids'][i]) == len(para_probs[i])
                    for p_idx, (p_id, p_prob) in enumerate(zip(batch['context_ids'][i], para_probs[i].tolist())):
                        p_title = self.env.get(p_id)['title']
                        if p_id not in self.memory[g_id]:
                            if p_idx == 0:
                                assert observations[len(next_commands)] in [None, p_id]
                                self.obs_scores[g_id].append(p_prob)
                            else:
                                assert review
                            if p_prob > para_threshold[i]:
                                self.memory[g_id][p_id] = p_prob
                                if gold_sp is not None and p_title not in gold_sp:
                                    logger.debug(f"{g_id}: add distractor {p_id} ({p_title}) for "
                                                 f"{p_prob:.4f} > {para_threshold[i]:.4f}")
                                    self.cases['add_dis'].add(g_id)
                            else:
                                if gold_sp is not None and p_title in gold_sp:
                                    logger.info(f"{g_id}: miss SP{len(self.memory[g_id])} {p_id} ({p_title}) for "
                                                f"{p_prob:.4f} <= {para_threshold[i]:.4f} "
                                                f"when {len(nn_input['input_ids'][i])} tokens")
                                    self.cases['miss_sp'].add(g_id)
                                    miss_sp = True
                        elif len(nn_input['input_ids'][i]) < 512 or p_idx < len(para_probs[i]) - 1:  # and not review:
                            alpha = 1.0
                            last_prob = self.memory[g_id][p_id]
                            self.memory[g_id][p_id] = (1.0 - alpha) * last_prob + alpha * p_prob
                            if self.memory[g_id][p_id] <= para_threshold[i]:
                                curr_prob = self.memory[g_id].pop(p_id)
                                self.discarded[g_id].append(p_id)
                                if gold_sp is not None and p_title in gold_sp:
                                    logger.info(f"{g_id}: remove SP{len(self.memory[g_id])} {p_id} ({p_title}) "
                                                f"from memory for {last_prob:.4f} -> {curr_prob:.4f} "
                                                f"<= {para_threshold[i]:.4f} "
                                                f"when {len(nn_input['input_ids'][i])} tokens")
                                    self.cases['rm_sp'].add(g_id)

                    # answer prediction
                    pred_ans = predict_answer(pred_start[i], pred_end[i], nn_input['input_ids'][i].tolist(),
                                              batch['context'][i], batch['context_token_spans'][i],
                                              nn_input['context_token_offset'][i], self.tokenizer)

                    # link target prediction
                    pred_target = batch['link_targets'][i][pred_link[i]] if pred_link[i] != 0 else 'nolink'

                    # score each proposal
                    proposals = []
                    for func_name, action_prob in zip(FUNCTIONS, action_probs[i].tolist()):
                        if func_name == 'ANSWER':
                            command = (func_name, pred_ans)
                            arg_conf = ans_conf[i].item()
                        elif func_name == 'BM25':
                            command = (func_name, batch['sparse_query'][i])
                            arg_conf = 1.
                        elif func_name == 'MDR':
                            if pred_exp[i] < len(para_probs[i]):
                                command = (func_name, (batch['question'][i], batch['context_ids'][i][pred_exp[i]]))
                            else:
                                command = (func_name, (batch['question'][i], None))
                            arg_conf = exp_conf[i].item()
                        else:
                            assert func_name == 'LINK'
                            command = (func_name, pred_target)
                            arg_conf = link_conf[i].item()
                        proposals.append((command, arg_conf, action_prob))
                    self.proposals[g_id].append(proposals)

                    # choose next_cmd
                    next_cmd = None
                    cmd_rank = 0
                    for command, arg_conf, action_prob in sorted(proposals, key=lambda x: -x[2]):
                        cmd_rank += 1
                        if command[0] in self.masked_functions:
                            continue
                        if command in self.exhausted_cmds[g_id]:
                            logger.debug(f"{g_id}: ignore exhausted command: {self.pretty_cmd(command)}")
                            continue
                        if self.oracle_belief and command[0] == 'ANSWER' and len(self.memory[g_id]) != 2:
                            logger.debug(f"{g_id}: give up rushing to answer "
                                         f"when {len(nn_input['input_ids'][i])} tokens in {len(self.memory[g_id])} SPs")
                            self.cases['early_ans'].add(g_id)
                            continue
                        if command[0] == 'ANSWER' and (command[1] == 'noanswer' or
                                                       len(para_probs[i]) < 2):
                            if command[1] == 'noanswer':
                                logger.debug(f"{g_id}: ignore ANSWER(none) when {len(nn_input['input_ids'][i])} tokens")
                                self.cases['ans_none'].add(g_id)
                            if len(para_probs[i]) < 2:  # or len(self.memory[g_id]) < 1:
                                logger.debug(f"{g_id}: ignore early ANSWER({command[1]}) "
                                             f"when {len(nn_input['input_ids'][i])} tokens")
                                self.cases['early_ans'].add(g_id)
                            continue
                        if command[0] == 'LINK' and command[1] == 'nolink':
                            logger.debug(f"{g_id}: ignore LINK(none)")
                            self.cases['link_none'].add(g_id)
                            continue
                        next_cmd = command
                        break
                    # finetune next_cmd
                    if next_cmd is None:
                        assert cmd_rank == 4
                        if 'BM25' not in self.masked_functions:
                            next_cmd = ('BM25', batch['question'][i])
                        elif 'MDR' not in self.masked_functions:
                            next_cmd = ('MDR', (batch['question'][i], None))
                        else:
                            next_cmd = ('ANSWER', 'noanswer')
                        logger.info(f"{g_id}: no competent command, use {self.pretty_cmd(next_cmd)} instead")
                    elif next_cmd[0] == 'ANSWER' and cmd_rank == 4:
                        if len(self.commands[g_id]) > 0 and self.commands[g_id][-1] not in self.exhausted_cmds[g_id]:
                            logger.info(f"{g_id}: replace the last resort {self.pretty_cmd(next_cmd)} "
                                        f"with the last command: {self.pretty_cmd(self.commands[g_id][-1])}")
                            next_cmd = self.commands[g_id][-1]
                        else:
                            logger.info(f"{g_id}: replace the last resort {self.pretty_cmd(next_cmd)} "
                                        f"with the naive command: BM25(question)")
                            next_cmd = ('BM25', batch['question'][i])
                        self.cases['bad_proposals'].add(g_id)
                    if self.oracle_state2action:
                        try:
                            # gold_links = set(batch['link_targets'][i]) & set(unescape(t) for t in gold_sp.keys())
                            if pred_target in set(unescape(t) for t in gold_sp.keys()):
                                next_cmd = ('LINK', pred_target)
                            elif len(self.memory[g_id]) == 0:
                                state2action = self.oracle_state2action[g_id]['initial']
                                func_name = min(state2action['sp_ranks'].keys(),
                                                key=lambda k: min(state2action['sp_ranks'][k].values()))
                                if func_name.startswith('BM25'):
                                    next_cmd = ('BM25', state2action['query'])
                                else:
                                    next_cmd = ('MDR', (batch['question'][i], None))
                            elif len(self.memory[g_id]) == 1:
                                sp1_id = list(self.memory[g_id].keys())[0]
                                state2action = self.oracle_state2action[g_id][unescape(self.env.get(sp1_id)['title'])]
                                func_name = min(state2action['sp2_ranks'].keys(),
                                                key=lambda k: state2action['sp2_ranks'][k])
                                if func_name.startswith('BM25'):
                                    next_cmd = ('BM25', state2action['query'])
                                else:
                                    next_cmd = ('MDR', (batch['question'][i], sp1_id))
                            else:
                                next_cmd = ('ANSWER', pred_ans)
                        except Exception as e:
                            print(e)
                            import pdb
                            pdb.set_trace()
                    elif self.oracle_belief and len(self.memory[g_id]) == 2:  # and pred_ans != 'noanswer':
                        next_cmd = ('ANSWER', pred_ans)
                    # else:
                    #     # if len(self.memory[g_id]) == 0:
                    #     #     next_cmd = ('BM25', batch['sparse_query'][i])
                    #     #     if next_cmd in self.exhausted_cmds[g_id]:
                    #     #         next_cmd = ('BM25', batch['question'][i])
                    #     #     if next_cmd in self.exhausted_cmds[g_id]:
                    #     #         next_cmd = ('ANSWER', 'noanswer')
                    #     # elif len(self.memory[g_id]) == 1:
                    #     #     next_cmd = ('LINK', pred_target)
                    #     #     if pred_target == 'nolink':
                    #     #         next_cmd = ('BM25', batch['question'][i])
                    #     #     # if pred_exp[i] < len(para_probs[i]):
                    #     #     #     next_cmd = ('MDR', (batch['question'][i], batch['context_ids'][i][pred_exp[i]]))
                    #     #     # else:
                    #     #     #     next_cmd = ('MDR', (batch['question'][i], None))
                    #     # else:
                    #     #     next_cmd = ('ANSWER', pred_ans)
                    #     opt_funcs = ['BM25', 'MDR']
                    #     if pred_target != 'nolink':
                    #         opt_funcs.append('LINK')
                    #     if pred_ans != 'noanswer':
                    #         opt_funcs.append('ANSWER')
                    #     options = [command for command, _, _ in proposals if command[0] in opt_funcs]
                    #     next_cmd = random.choice(options)
                    next_commands[g_id] = next_cmd
                    self.commands[g_id].append(next_cmd)

                    if pred_ans != 'noanswer':
                        norm_pred_ans = normalize_answer(pred_ans)
                        if norm_pred_ans not in self.all_answer[g_id]:
                            self.all_answer[g_id][norm_pred_ans] = 0.
                        self.all_answer[g_id][norm_pred_ans] += ans_conf[i].item()
                    for sent_idx, sent_prob in enumerate(sent_probs[i].tolist()):
                        sent_loc = tuple(batch['sents_map'][i][sent_idx])
                        if sent_prob > sent_threshold[i]:
                            self.all_sp_facts[g_id][sent_loc] = sent_prob
                        elif sent_loc in self.all_sp_facts[g_id]:
                            self.all_sp_facts[g_id].pop(sent_loc)

                    if next_cmd[0] == 'ANSWER':
                        # set answer prediction
                        self.answer[g_id] = next_cmd[1]
                        if gold_ans is not None:
                            if f1_score(self.answer[g_id], gold_ans)[0] >= 0.75:
                                self.cases['good_ans'].add(g_id)
                        if miss_sp:
                            self.cases['miss_sp2'].add(g_id)

                        # augment memory
                        ext_mem = []
                        for p_id, p_span in zip(batch['context_ids'][i], nn_input['paras_span'][i]):
                            if self.oracle_belief:
                                break
                            if pred_start[i] in [1, 2, NA_POS]:
                                break
                            if p_span[0] <= pred_start[i] <= pred_end[i] <= p_span[1]:
                                if p_id not in self.memory[g_id] and (not review or len(self.memory[g_id]) < 2):
                                    self.memory[g_id][p_id] = 1.
                                    ext_mem.append(p_id)
                                break
                        for (p_title, s_idx), s_prob in sorted(self.all_sp_facts[g_id].items(), key=lambda x: -x[1]):
                            if self.oracle_belief:
                                break
                            p_prob_ = s_prob * 0.5
                            if len(self.memory[g_id]) >= 2 and p_prob_ <= sorted(self.memory[g_id].values())[-2]:
                                break
                            p_id = self.env.title2id(unescape(p_title))
                            if p_id not in self.memory[g_id]:  # and p_id in batch['context_ids'][i]:
                                self.memory[g_id][p_id] = p_prob_
                                ext_mem.append(p_id)
                        for p_id, p_prob in sorted(zip(self.observed[g_id], self.obs_scores[g_id]),
                                                   key=lambda x: -x[1]):
                            if self.oracle_belief:
                                break
                            p_prob_ = p_prob * 0.4
                            if len(self.memory[g_id]) >= 2 and p_prob_ <= sorted(self.memory[g_id].values())[-2]:
                                break
                            if p_id not in self.memory[g_id]:
                                self.memory[g_id][p_id] = p_prob_
                                ext_mem.append(p_id)

                        # set supporting passage prediction
                        sp_ids = sorted(self.memory[g_id].keys(), key=lambda x: -self.memory[g_id][x])[:2]
                        if gold_sp is not None and len(ext_mem) > 0:
                            logger.info(f"{g_id}: augment SP {ext_mem} into memory")
                            for p_id in sp_ids:
                                if p_id in ext_mem and self.env.get(p_id)['title'] not in gold_sp:
                                    logger.warning(
                                        f"{g_id}: augment false SP {p_id} into memory {self.memory[g_id]}")
                                    self.cases['aug_false_spp'].add(g_id)
                        sp_titles = [self.env.get(p_id)['title'] for p_id in sp_ids]
                        self.sp_passages[g_id] = sp_titles
                        # set supporting passage prediction
                        sp_title_set = set()
                        sorted_sent_probs, sent_indices = sent_probs[i].sort(descending=True)
                        for sent_prob, sent_idx in zip(sorted_sent_probs.tolist(), sent_indices.tolist()):
                            if sent_prob <= sent_threshold[i] and len(sp_title_set) == len(sp_titles):
                                break
                            sent_loc = tuple(batch['sents_map'][i][sent_idx])
                            if sent_loc[0] in sp_titles:
                                if sent_prob > sent_threshold[i] or sent_loc[0] not in sp_title_set:
                                    self.sp_facts[g_id][sent_loc] = sent_prob
                                    sp_title_set.add(sent_loc[0])
                        if gold_sp is not None:
                            if sp_title_set == set(gold_sp.keys()):
                                self.cases['good_spp'].add(g_id)
                                if set(self.sp_facts[g_id].keys()) != set([(t, s_idx) for t, sents in gold_sp.items()
                                                                           for s_idx in sents]):
                                    self.cases['bad_sp'].add(g_id)
                    else:
                        if gold_ans is not None and f1_score(pred_ans, gold_ans)[0] >= 0.60:
                            logger.debug(f"{g_id}: decide to {next_cmd[0]} but miss ans "
                                         f"{ans_conf[i]:.2f} 『{pred_ans}』 = 『{gold_ans}』 "
                                         f"when {len(para_probs[i])}/{len(self.memory[g_id])} passages")
                            self.cases['miss_ans'].add(g_id)
                        if (next_cmd[0] == 'MDR' and next_cmd[1][1] is not None and
                                gold_sp is not None and self.env.get(next_cmd[1][1])['title'] not in gold_sp):
                            logger.debug(f"{g_id}: false dense query expansion {next_cmd[1][1]} "
                                         f"({self.env.get(next_cmd[1][1])['title']}) not in {list(gold_sp.keys())}")
                            self.cases['false_expansion'].add(g_id)

        assert len(next_commands) == len(game_ids)

        return next_commands

    def force_answer(self, game_ids, questions):
        samples = []
        for g_id, question in zip(game_ids, questions):
            tmp_mem = copy.deepcopy(self.memory[g_id])
            for (p_title, s_idx), s_prob in sorted(self.all_sp_facts[g_id].items(), key=lambda x: -x[1]):
                if len(tmp_mem) > 5:
                    break
                p_id = self.env.title2id(unescape(p_title))
                if p_id not in tmp_mem:
                    tmp_mem[p_id] = s_prob * 0.5
            for p_id, p_prob in sorted(zip(self.observed[g_id], self.obs_scores[g_id]), key=lambda x: -x[1]):
                if len(tmp_mem) > 5:
                    break
                if p_id not in tmp_mem:
                    tmp_mem[p_id] = p_prob * 0.75
            context_ids = sorted(tmp_mem.keys(), key=lambda x: -tmp_mem[x]) if len(tmp_mem) > 0 else [None]
            samples.append({
                "q_id": g_id,
                "question": question,
                "context_ids": context_ids,
                "passages": [self.env.get(p_id) for p_id in context_ids],
                "sparse_query": question,
                "excluded": set(self.clicked[g_id])
            })

        dataset = ActionDataset(samples, self.tokenizer, self.env.title2id,
                                max_seq_len=self.max_seq_len, max_q_len=self.max_q_len, max_obs_len=self.max_obs_len,
                                strict=self.strict)
        collate_func = partial(collate_transitions, pad_id=self.tokenizer.pad_token_id)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_func,
                                pin_memory=True, num_workers=0)

        answers, _answers, sp_passages, sp_facts, _sp_facts = {}, {}, {}, {}, {}
        self.union_model.eval()
        for batch in tqdm(dataloader, desc='act', total=len(dataloader), disable=len(dataloader) <= 100):
            nn_input = to_device(batch['nn_input'], self.device)
            with torch.no_grad():
                if self.oracle_belief:
                    oracle_para_logits = [
                        torch.tensor([20. * float(self.env.get(p_id)['title'] in self.gold_qas_map[g_id][2]) - 10.
                                      for p_id in p_ids], device=self.device)
                        for p_ids in batch['context_ids']
                    ]
                    outputs = self.union_model(nn_input, oracle_para_logits, top_k=2)
                else:
                    outputs = self.union_model(nn_input, top_k=2)

                # (B, _P)    (B, _S)      (B,)            (B,)
                para_logits, sent_logits, para_threshold, sent_threshold = outputs[4:8]
                # (B,)       (B, 2)     (B, 2)      (B,)       (B,)
                pred_action, top_start, top_end, pred_link, pred_exp = outputs[8:13]
                # (B,)    (B,)       (B,)
                ans_conf, link_conf, exp_conf = outputs[13:]

                para_probs = [_para_logits.sigmoid() for _para_logits in para_logits]  # (B, _P)
                sent_probs = [_sent_logits.sigmoid() for _sent_logits in sent_logits]  # (B, _S)
                para_threshold = para_threshold.sigmoid()  # (B,)
                sent_threshold = sent_threshold.sigmoid()  # (B,)

                for i, g_id in enumerate(batch['q_id']):
                    if self.gold_qas_map is None:
                        gold_ans, gold_sp = None, None
                    else:
                        gold_ans, gold_sp = self.gold_qas_map.get(g_id, (None, None, None))[1:]

                    tmp_all_answer = copy.deepcopy(self.all_answer[g_id])
                    tmp_all_sp_facts = copy.deepcopy(self.all_sp_facts[g_id])
                    tmp_mem = copy.deepcopy(self.memory[g_id])

                    # answer prediction
                    pred_ans = predict_answer(top_start[i][0], top_end[i][0], nn_input['input_ids'][i].tolist(),
                                              batch['context'][i], batch['context_token_spans'][i],
                                              nn_input['context_token_offset'][i], self.tokenizer)
                    pred_ans_ = predict_answer(top_start[i][1], top_end[i][1], nn_input['input_ids'][i].tolist(),
                                               batch['context'][i], batch['context_token_spans'][i],
                                               nn_input['context_token_offset'][i], self.tokenizer)

                    # update temporary all norm answers
                    if pred_ans != 'noanswer':
                        norm_pred_ans = normalize_answer(pred_ans)
                        if norm_pred_ans not in tmp_all_answer:
                            tmp_all_answer[norm_pred_ans] = 0.
                        tmp_all_answer[norm_pred_ans] += ans_conf[i].item()

                    # update temporary all sp sentences
                    for sent_idx, sent_prob in enumerate(sent_probs[i].tolist()):
                        sent_loc = tuple(batch['sents_map'][i][sent_idx])
                        if sent_prob > sent_threshold[i]:
                            tmp_all_sp_facts[sent_loc] = sent_prob
                        elif sent_loc in tmp_all_sp_facts:
                            tmp_all_sp_facts.pop(sent_loc)

                    # update temporary memory
                    assert len(batch['context_ids'][i]) == len(para_probs[i])
                    for p_idx, (p_id, p_prob) in enumerate(zip(batch['context_ids'][i], para_probs[i].tolist())):
                        p_title = self.env.get(p_id)['title']
                        if p_id not in tmp_mem:
                            if p_prob > para_threshold[i]:
                                tmp_mem[p_id] = p_prob
                                if gold_sp is not None and p_title not in gold_sp:
                                    logger.debug(f"{g_id}: add distractor {p_id} ({p_title}) for "
                                                 f"{p_prob:.4f} > {para_threshold[i]:.4f}")
                            else:
                                if gold_sp is not None and p_title in gold_sp:
                                    logger.debug(f"{g_id}: miss SP{len(tmp_mem)} {p_id} ({p_title}) for "
                                                 f"{p_prob:.4f} <= {para_threshold[i]:.4f} "
                                                 f"when {len(nn_input['input_ids'][i])} tokens")
                        else:
                            alpha = 1.0
                            last_prob = tmp_mem[p_id]
                            if len(nn_input['input_ids'][i]) < 512 or p_idx < len(para_probs[i]) - 1:
                                tmp_mem[p_id] = (1.0 - alpha) * last_prob + alpha * p_prob
                            if tmp_mem[p_id] <= para_threshold[i]:
                                curr_prob = tmp_mem.pop(p_id)
                                if gold_sp is not None and p_title in gold_sp:
                                    logger.debug(f"{g_id}: remove SP{len(tmp_mem)} {p_id} ({p_title}) from "
                                                 f"memory for {last_prob:.4f} -> {curr_prob:.4f} "
                                                 f"<= {para_threshold[i]:.4f} "
                                                 f"when {len(nn_input['input_ids'][i])} tokens")

                    # augment temporary memory
                    ext_mem = []
                    for p_id, p_span in zip(batch['context_ids'][i], nn_input['paras_span'][i]):
                        if self.oracle_belief:
                            break
                        if top_start[i][0] in [1, 2, NA_POS]:
                            break
                        if p_span[0] <= top_start[i][0] <= top_end[i][0] <= p_span[1]:
                            if p_id not in tmp_mem:
                                tmp_mem[p_id] = 1.
                                ext_mem.append(p_id)
                            break
                    for (p_title, s_idx), s_prob in sorted(tmp_all_sp_facts.items(), key=lambda x: -x[1]):
                        if self.oracle_belief:
                            break
                        p_prob_ = s_prob * 0.5
                        if len(tmp_mem) >= 2 and p_prob_ <= sorted(tmp_mem.values())[-2]:
                            break
                        p_id = self.env.title2id(unescape(p_title))
                        if p_id not in tmp_mem:  # and p_id in batch['context_ids'][i]:
                            tmp_mem[p_id] = p_prob_
                            ext_mem.append(p_id)
                    for p_id, p_prob in sorted(zip(self.observed[g_id], self.obs_scores[g_id]), key=lambda x: -x[1]):
                        if self.oracle_belief:
                            break
                        p_prob_ = p_prob * 0.4
                        if len(tmp_mem) >= 2 and p_prob_ <= sorted(tmp_mem.values())[-2]:
                            break
                        if p_id not in tmp_mem:
                            tmp_mem[p_id] = p_prob_
                            ext_mem.append(p_id)
                    sp_ids = sorted(tmp_mem.keys(), key=lambda x: -tmp_mem[x])[:2]
                    if gold_sp is not None and len(ext_mem) > 0:
                        logger.debug(f"{g_id}: augment SP {ext_mem} into memory")
                        for p_id in sp_ids:
                            if p_id in ext_mem and self.env.get(p_id)['title'] not in gold_sp:
                                logger.debug(f"{g_id}: augment false SP {p_id} into memory {tmp_mem}")

                    _answers[g_id] = max(tmp_all_answer.keys(),
                                         key=lambda x: tmp_all_answer[x]) if len(tmp_all_answer) > 0 else pred_ans_
                    answers[g_id] = pred_ans if pred_ans != 'noanswer' else _answers[g_id]  # pred_ans_  #
                    sp_passages[g_id] = [self.env.get(p_id)['title'] for p_id in sp_ids]
                    sp_facts[g_id] = []
                    sp_title_set = set()
                    sorted_sent_probs, sent_indices = sent_probs[i].sort(descending=True)
                    for sent_prob, sent_idx in zip(sorted_sent_probs.tolist(), sent_indices.tolist()):
                        if sent_prob <= sent_threshold[i] and len(sp_title_set) == len(sp_passages[g_id]):
                            break
                        sent_loc = tuple(batch['sents_map'][i][sent_idx])
                        if sent_loc[0] in sp_passages[g_id]:
                            if sent_prob > sent_threshold[i] or sent_loc[0] not in sp_title_set:
                                sp_facts[g_id].append(sent_loc)
                                sp_title_set.add(sent_loc[0])
                    _sp_facts[g_id] = list(tmp_all_sp_facts.keys())

        return answers, _answers, sp_passages, sp_facts, _sp_facts
