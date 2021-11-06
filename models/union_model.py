# import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModel  # , AutoConfig

from config import FUNCTIONS, NA_POS
from utils.tensor_utils import mask_pos0  # , pad_tensors
from utils.rank_losses import list_mle  # , pairwise_hinge  # , list_net
from utils.utils import span_f1


def compute_spans_loss(start_logits, end_logits, start_positions, end_positions):
    loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-1)

    # compute span loss
    # (A, B)
    start_losses = [loss_fct(start_logits, _start_positions) for _start_positions in start_positions.unbind(dim=1)]
    end_losses = [loss_fct(end_logits, _end_positions) for _end_positions in end_positions.unbind(dim=1)]
    span_losses = torch.stack(start_losses, dim=1) + torch.stack(end_losses, dim=1)  # (B, A)
    span_loss = calc_mml(span_losses)
    return span_loss


def calc_mml(losses):
    log_probs = -losses  # (B, A)
    log_probs = mask_pos0(log_probs, (log_probs != 0).float())  # (B, A)
    # log_probs = log_probs.masked_fill(log_probs == 0, float('-inf'))
    marginal_prob = torch.exp(log_probs).sum(dim=1)  # (B,)
    return -torch.log(marginal_prob + (marginal_prob == 0).float())  # (B,)


def get_paras_weight(paras_loss, obs_idx=0, obs_weight=0.8):
    if paras_loss.size(0) == 0:
        paras_weight = torch.zeros_like(paras_loss)
    elif paras_loss.size(0) == 1:
        paras_weight = torch.ones_like(paras_loss)
    elif obs_weight > 0.0:
        paras_weight = torch.full_like(paras_loss, (1 - obs_weight) / (paras_loss.size(0) - 1))
        paras_weight[obs_idx] = obs_weight
        assert paras_weight.sum() == 1
    else:
        paras_weight = torch.full_like(paras_loss, 1 / paras_loss.size(0))

    return paras_weight


class UnionModel(nn.Module):

    def __init__(self, encoder_name, max_ans_len, sp_weight=0.05):
        super(UnionModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.reranker = nn.Linear(self.hidden_size, 1)
        self.sp_cls = nn.Linear(self.hidden_size, 1)

        self.answerer = Answerer(self.hidden_size, max_ans_len)
        self.linker = Linker(self.hidden_size)
        # self.sparse_generator = nn.Linear(self.hidden_size, 1)
        # self.dense_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        self.commander = Commander(self.hidden_size, hidden_dropout_prob=0.1)

        self.bce_loss = BCEWithLogitsLoss(reduction='none')
        self.ce_loss = CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.sp_weight = sp_weight

    def forward(self, batch, oracle_para_logits=None, top_k=1):
        # (B, T, H)
        seq_hiddens = self.encoder(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids', None))[0]
        # seq_hiddens[:, 3].zero_()

        # sep_index = batch['context_token_offset'] - 1  # (B,)
        # sep_hidden = seq_hiddens.gather(
        #     1, sep_index[:, None, None].expand(-1, -1, self.hidden_size)
        # ).squeeze(1)  # (B, H)
        # question_hidden = torch.stack([  # (B, H)
        #     seq_hiddens[i, 4:sep_idx].mean(0) for i, sep_idx in enumerate(sep_index)
        # ], dim=0)

        none_hidden = seq_hiddens[:, 3, :]  # (B, H)
        para_threshold = self.reranker(none_hidden).squeeze(-1)  # (B,)
        # para_threshold = seq_hiddens.new_full(seq_hiddens.size()[:1], 0.0)  # (B,)
        # sent_threshold = self.sp_cls(question_hidden).squeeze(-1)  # (B,)
        sent_threshold = seq_hiddens.new_full(seq_hiddens.size()[:1], 0.0)  # (B,)

        paras_mark = batch['paras_mark']  # (B, _P)
        # paras_mark = [[end + 1 for start, end in _paras_span] for _paras_span in batch['paras_span']]  # (B, _P)
        para_num = [len(_paras_mark) for _paras_mark in paras_mark]  # (B,)
        para_hiddens = [seq_hiddens[i, _paras_mark] for i, _paras_mark in enumerate(paras_mark)]  # (B, _P, H)
        # para_hiddens = [  # (B, _P, H)
        #     torch.stack([
        #         seq_hiddens[i, start:end + 1].mean(0) for (start, end) in _paras_span
        #     ], dim=0) if len(_paras_span) > 0 else seq_hiddens.new_empty((0, self.hidden_size))
        #     for i, _paras_span in enumerate(batch['paras_span'])
        # ]
        if oracle_para_logits is None:
            para_logits = self.reranker(torch.cat(para_hiddens, dim=0)).squeeze(-1).split(para_num, dim=0)  # (B, _P)
        else:
            para_logits = oracle_para_logits  # (B, _P)

        sent_num = [len(sents_span) for sents_span in batch['sents_span']]  # (B,)
        sent_hiddens = [  # (B, _S, H)
            torch.stack([
                seq_hiddens[i, start:end + 1].mean(0) for (start, end) in sents_span
            ], dim=0) if len(sents_span) > 0 else seq_hiddens.new_empty((0, self.hidden_size))
            for i, sents_span in enumerate(batch['sents_span'])
        ]
        sent_logits = self.sp_cls(torch.cat(sent_hiddens, dim=0)).squeeze(-1).split(sent_num, dim=0)  # (B, _S)

        # (B, T)      (B, T)      (B, *)     (B, *)
        start_logits, end_logits, top_start, top_end = self.answerer(seq_hiddens, batch['answer_mask'], top_k)
        # (B, T)     (B, T)
        start_probs, end_probs = start_logits.softmax(dim=-1), end_logits.softmax(dim=-1)
        # (B,)
        if top_start.dim() > 1:
            pred_start, pred_end = top_start[:, 0], top_end[:, 0]
        else:
            pred_start, pred_end = top_start, top_end
        ans_conf = (start_probs.gather(1, pred_start[:, None]) * end_probs.gather(1, pred_end[:, None])).squeeze(1)
        # (B, H)
        answer_hidden = torch.stack([seq_hiddens[i, start:end + 1].mean(0)
                                     for i, (start, end) in enumerate(zip(pred_start, pred_end))])
        answer_hidden *= (ans_conf / ans_conf.detach()).unsqueeze(1)

        # (B, H)
        sparse_hidden = torch.stack([seq_hiddens[i, start:end + 1].mean(0)
                                     for i, (start, end) in enumerate(zip(batch['sparse_start'], batch['sparse_end']))])

        pred_exp, exp_conf = [], []  # (B,)
        dense_hidden = torch.empty_like(sparse_hidden)  # (B, H)
        for i, _para_th in enumerate(para_threshold):
            _para_logits_ = torch.cat([para_logits[i], _para_th[None]], dim=0)
            _exp_conf, _exp_idx = _para_logits_.softmax(0).max(dim=0)
            pred_exp.append(_exp_idx)
            exp_conf.append(_exp_conf)
            if _exp_idx < len(para_logits[i]):
                # _exp_start, _exp_end = batch['paras_span'][i][_exp_idx]
                # dense_hidden[i] = seq_hiddens[i, _exp_start:_exp_end + 1].mean(0)  # * _exp_conf
                dense_hidden[i] = para_hiddens[i][_exp_idx]  # * _exp_conf
            else:
                dense_hidden[i] = none_hidden[i]  # * _exp_conf
            dense_hidden[i] *= _exp_conf / _exp_conf.detach()
            '''
            if len(para_hiddens[i]) > 0:
                # if self.training:
                #     gold_exp = batch['dense_expansion'][i]
                #     if 0 <= gold_exp < len(para_hiddens[i]):
                #         dense_hidden[i] += para_hiddens[i][gold_exp] * para_logits[i][gold_exp].sigmoid()
                # dense_hidden[i] += torch.matmul((para_logits[i] > _para_th).float(), para_hiddens[i])
                # dense_hidden[i] += torch.matmul(
                #     para_logits[i].sigmoid().where(para_logits[i] > _para_th, torch.zeros_like(para_logits[i])),
                #     para_hiddens[i]
                # )
                _exp_logit, _exp_idx = para_logits[i].max(dim=0)
                if _exp_logit > _para_th:
                    dense_hidden[i] += para_hiddens[i][_exp_idx] * _exp_logit.sigmoid()
            #     else:
            #         dense_hidden[i] += none_hidden[i] * _para_th.sigmoid()
            # else:
            #     dense_hidden[i] += none_hidden[i] * _para_th.sigmoid()
            '''
        pred_exp = torch.stack(pred_exp)
        exp_conf = torch.stack(exp_conf)

        # (B, _L)    (B, _L, H)    (B,)
        link_logits, link_hiddens, pred_link = self.linker(seq_hiddens, batch['links_spans'])
        link_probs = [_link_logits.softmax(dim=-1) for _link_logits in link_logits]  # (B, _L)
        link_conf = torch.stack([link_probs[i][idx] for i, idx in enumerate(pred_link)])  # (B,)
        # (B, H)
        link_hidden = torch.stack([link_hiddens[i][idx] for i, idx in enumerate(pred_link)])
        link_hidden *= (link_conf / link_conf.detach()).unsqueeze(1)

        state_hidden = seq_hiddens[:, 0, :]  # (B, H)

        # (B, 4)       (B,)
        action_logits, pred_action = self.commander(state_hidden,
                                                    answer_hidden, sparse_hidden, dense_hidden, link_hidden)

        if self.training:
            for i, action_label in enumerate(batch['action_label']):  # adjust/ignore action_label dynamically
                if action_label == -1:
                    continue
                tgt_spans = [(s, e) for s, e in zip(batch['answer_starts'][i], batch['answer_ends'][i])
                             if -1 not in (s, e)]
                # gold_exp = batch['dense_expansion'][i]
                if FUNCTIONS[action_label] == 'LINK' and pred_link[i] != batch['link_label'][i]:
                    batch['action_label'][i] = -1  # batch['ranked_action_labels'][i][1] # avoid wrong/meaningless link
                elif (FUNCTIONS[action_label] == 'ANSWER' and
                      # pred_start[i] not in batch['answer_starts'][i] and pred_end[i] not in batch['answer_ends'][i]):
                      max(span_f1((pred_start[i], pred_end[i]), tgt_span) for tgt_span in tgt_spans) < 0.4):
                    batch['action_label'][i] = -1  # avoid misleading Commander to answer wrong
                # elif (self.commander.actions[action_label] == 'MDR' and
                #       ((gold_exp < 0 and pred_exp[i] != len(para_logits[i])) or 0 <= gold_exp != pred_exp[i])):
                #     batch['action_label'][i] = -1
            action_loss = self.ce_loss(action_logits, batch['action_label'])  # (B,)

            # (A, B)
            start_losses = [self.ce_loss(start_logits, ans_start) for ans_start in batch['answer_starts'].unbind(dim=1)]
            end_losses = [self.ce_loss(end_logits, ans_end) for ans_end in batch['answer_ends'].unbind(dim=1)]
            ans_losses = torch.stack(start_losses, dim=1) + torch.stack(end_losses, dim=1)  # (B, A)
            ans_loss = calc_mml(ans_losses)  # (B,)

            padded_link_logits = pad_sequence(link_logits, batch_first=True, padding_value=-10000.0)  # (B, L)
            link_loss = self.ce_loss(padded_link_logits, batch['link_label'])  # (B,)

            para_rank_loss = torch.zeros_like(link_loss)  # (B,)
            for i, (_para_logits, _paras_label) in enumerate(zip(para_logits, batch['paras_label'])):
                _para_logits = torch.cat([para_threshold[i:i + 1], _para_logits], dim=0)  # (1 + _P,)
                _paras_label = torch.cat([_paras_label.new_tensor([0.5]), _paras_label], dim=0)  # (1 + _P,)
                if _paras_label.size(0) > 1 and _paras_label.max() != _paras_label.min():
                    para_rank_loss[i] = list_mle(_para_logits.unsqueeze(0), _paras_label.unsqueeze(0))
                    # para_rank_loss[i] = list_net(_para_logits[None], _paras_label[None], irrelevant_val=0.)
                    # para_rank_loss[i] = pairwise_hinge(_para_logits.unsqueeze(0), _paras_label.unsqueeze(0))
            # sent_rank_loss = torch.zeros_like(link_loss)  # (B,)
            # for i, (_sent_logits, _sents_label) in enumerate(zip(sent_logits, batch['sents_label'])):
            #     _sent_logits = torch.cat([sent_threshold[i:i + 1], _sent_logits], dim=0)  # (1 + _P,)
            #     _sents_label = torch.cat([_sents_label.new_tensor([0.5]), _sents_label], dim=0)  # (1 + _P,)
            #     if _sents_label.size(0) > 1 and _sents_label.max() != _sents_label.min():
            #         # sent_rank_loss[i] = list_mle(_sent_logits.unsqueeze(0), _sents_label.unsqueeze(0))
            #         # sent_rank_loss[i] = list_net(_sent_logits[None], _sents_label[None], irrelevant_val=0.)
            #         sent_rank_loss[i] = pairwise_hinge(_sent_logits.unsqueeze(0), _sents_label.unsqueeze(0))

            # # (B, _P)
            # paras_loss = self.bce_loss(torch.cat(para_logits), torch.cat(batch['paras_label'])).split(para_num, dim=0)
            # para_loss = torch.stack([(_paras_loss * get_paras_weight(_paras_loss, obs_weight=-1)).sum()  # (B,)
            #                          for _paras_loss in paras_loss], dim=0)
            # (B, _S)
            sents_loss = self.bce_loss(torch.cat(sent_logits), torch.cat(batch['sents_label'])).split(sent_num, dim=0)
            sent_loss = torch.stack([_sents_loss.mean() if _sents_loss.size(0) > 0 else _sents_loss.sum()  # (B,)
                                     for _sents_loss in sents_loss], dim=0)

            losses = {
                "all": (
                        action_loss + ans_loss + link_loss +
                        para_rank_loss +  # self.sp_weight * sent_rank_loss + para_loss +
                        self.sp_weight * sent_loss
                ).mean(),
                "action": action_loss.mean().detach(),
                "answer": ans_loss.mean().detach(),
                "link": link_loss.mean().detach(),
                "memory": para_rank_loss.mean().detach(),
                # "sent_rank": sent_rank_loss.mean().detach(),
                # "para": para_loss.mean().detach(),
                "sent": sent_loss.mean().detach()
            }

            return losses

        #       (B, 4)         (B, T)        (B, T)      (B, _L)      (B, _P)      (B, _S)
        return (action_logits, start_logits, end_logits, link_logits, para_logits, sent_logits,
                # (B,)          (B,)
                para_threshold, sent_threshold,
                # (B,)       (B, *)     (B, *)      (B,)       (B,)
                pred_action, top_start, top_end, pred_link, pred_exp,
                # (B,)    (B,)       (B,)
                ans_conf, link_conf, exp_conf)


class Answerer(nn.Module):
    def __init__(self, hidden_size, max_ans_len):
        super(Answerer, self).__init__()
        self.hidden_size = hidden_size
        self.max_ans_len = max_ans_len

        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.span_mask = None

    def get_span_mask(self, seq_len, device):
        if self.span_mask is not None and seq_len <= self.span_mask.size(0):
            return self.span_mask[:seq_len, :seq_len].to(device)
        self.span_mask = torch.tril(torch.triu(torch.ones((seq_len, seq_len), device=device), 0), self.max_ans_len - 1)
        self.span_mask[:4, :] = 0
        self.span_mask[1, 1] = 1
        self.span_mask[2, 2] = 1
        self.span_mask[NA_POS, NA_POS] = 1
        return self.span_mask

    def forward(self, seq_hiddens, ans_mask, top_k=1):
        """

        Args:
            seq_hiddens (torch.FloatTensor): (B, S, H)
            ans_mask (torch.FloatTensor):  (B, S)
            top_k (int):

        Returns:
            tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor]
        """
        batch_size, seq_len = ans_mask.size()

        # (B, S, 2)
        span_logits = self.qa_outputs(seq_hiddens)
        masked_span_logits = mask_pos0(span_logits, ans_mask.unsqueeze(dim=-1))
        start_logits, end_logits = masked_span_logits.unbind(dim=-1)  # (B, S)

        # (B, S, S)
        span_scores = start_logits[:, :, None] + end_logits[:, None]
        span_mask = self.get_span_mask(seq_len, start_logits.device).unsqueeze(0)
        masked_span_scores = mask_pos0(span_scores, span_mask)
        # (B, *)
        top_span = masked_span_scores.view(batch_size, -1).argsort(dim=-1, descending=True)[:, :top_k].squeeze(-1)
        top_start = top_span // seq_len
        top_end = top_span % seq_len

        return start_logits, end_logits, top_start, top_end


class Linker(nn.Module):
    def __init__(self, hidden_size):
        super(Linker, self).__init__()
        self.hidden_size = hidden_size

        self.scorer = nn.Linear(hidden_size, 1)

    def forward(self, seq_hiddens, links_spans):
        """

        Args:
            seq_hiddens (torch.FloatTensor): (B, S, H)
            links_spans (list): (B, _L, _M, 2)

        Returns:
            tuple[list[torch.FloatTensor], list[torch.FloatTensor], torch.LongTensor]
        """
        link_num = [len(_links_spans) for _links_spans in links_spans]  # (B,)
        link_hiddens = [  # (B, _L, H)
            torch.stack([
                torch.stack([seq_hiddens[i, start:end + 1].mean(0) for (start, end) in link_spans], dim=0).max(0)[0]
                for link_spans in _links_spans
            ])  # if len(links_spans) > 0 else seq_hiddens.new_empty(0, self.hidden_size)
            for i, _links_spans in enumerate(links_spans)
        ]
        # (B, _L)
        link_logits = self.scorer(torch.cat(link_hiddens, dim=0)).squeeze(-1).split(link_num, dim=0)
        best_idx = torch.stack([_links_logits.argmax(dim=-1) for _links_logits in link_logits])  # (B,)

        return link_logits, link_hiddens, best_idx


class Commander(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob=0.0):
        super().__init__()
        self.hidden_size = hidden_size

        func_weights = torch.empty(len(FUNCTIONS), hidden_size)  # .normal_(mean=0.0, std=0.02)
        nn.init.xavier_uniform_(func_weights)
        self.func_embedding = nn.Parameter(func_weights, requires_grad=True)

        self.ffn = FFN(3 * hidden_size, hidden_size, intermediate_size=4 * hidden_size,  # None,  #
                       hidden_dropout_prob=hidden_dropout_prob)
        self.act_scorer = nn.Linear(hidden_size, 1)

    def forward(self, state_hidden, answer_hidden, sparse_hidden, dense_hidden, link_hidden):
        """

        Args:
            state_hidden (torch.FloatTensor): (B, H)
            answer_hidden (torch.FloatTensor): (B, H)
            sparse_hidden (torch.FloatTensor): (B, H)
            dense_hidden (torch.FloatTensor): (B, H)
            link_hidden (torch.FloatTensor): (B, H)

        Returns:
            tuple[torch.FloatTensor, torch.LongTensor]
        """
        arg_hiddens = torch.stack([answer_hidden, sparse_hidden, dense_hidden, link_hidden], dim=1)  # (B, 4, H)
        action_hiddens = torch.cat([self.func_embedding.unsqueeze(0).expand(arg_hiddens.size(0), -1, -1),
                                    arg_hiddens], dim=-1)  # (B, 4, 2H)
        # action_hiddens = self.func_embedding.unsqueeze(0).expand(arg_hiddens.size(0), -1, -1) + arg_hiddens
        # (B, 4, 3H)
        transition_hiddens = torch.cat([state_hidden.unsqueeze(1).expand(-1, 4, -1), action_hiddens], dim=-1)
        transition_hiddens = self.ffn(transition_hiddens)  # (B, 4, H)
        action_logits = self.act_scorer(transition_hiddens).squeeze(-1)  # (B, 4)
        best_idx = action_logits.argmax(dim=-1)  # (B,)

        return action_logits, best_idx


class FFN(nn.Module):
    def __init__(self, in_size, out_size, intermediate_size=None, hidden_dropout_prob=0.0):
        super().__init__()
        self.intermediate_size = intermediate_size
        if intermediate_size is not None:
            self.dense1 = nn.Linear(in_size, intermediate_size)
            self.dense2 = nn.Linear(intermediate_size, out_size)
        else:
            self.dense1 = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        output = F.gelu(self.dense1(x))
        if self.intermediate_size is not None:
            output = self.dense2(output)
        output = self.dropout(output)
        return output
