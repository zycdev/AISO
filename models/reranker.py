import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from transformers import AutoModel  # , AutoConfig

from utils.rank_losses import list_mle  # , list_net
from .union_model import get_paras_weight


class Reranker(nn.Module):

    def __init__(self, encoder_name):
        super(Reranker, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.reranker = nn.Linear(self.hidden_size, 1)

        self.bce_loss = BCEWithLogitsLoss(reduction='none')

    def forward(self, batch):
        # (B, T, H)
        seq_hiddens = self.encoder(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids', None))[0]

        para_num = [len(paras_mark) for paras_mark in batch['paras_mark']]  # (B,)
        para_hiddens = [seq_hiddens[i, paras_mark] for i, paras_mark in enumerate(batch['paras_mark'])]  # (B, _P, H)
        # (B, _P)
        para_logits = self.reranker(torch.cat(para_hiddens, dim=0)).squeeze(-1).split(para_num, dim=0)

        if self.training:
            # (B, _P)
            paras_loss = self.bce_loss(torch.cat(para_logits), torch.cat(batch['paras_label'])).split(para_num, dim=0)
            para_loss = torch.stack([(_paras_loss * get_paras_weight(_paras_loss, obs_weight=-1)).sum()
                                     for _paras_loss in paras_loss], dim=0)  # (B,)

            memory_loss = torch.zeros_like(para_loss)  # (B,)
            for i, (_para_logits, _paras_label) in enumerate(zip(para_logits, batch['paras_label'])):
                if _paras_label.size(0) > 1 and _paras_label.max() != _paras_label.min():
                    memory_loss[i] = list_mle(_para_logits.unsqueeze(0), _paras_label.unsqueeze(0))
                    # memory_loss[i] = list_net(_para_logits.unsqueeze(0), _paras_label.unsqueeze(0), irrelevant_val=0.)

            loss = (para_loss + memory_loss).mean()

            return loss

        return para_logits
