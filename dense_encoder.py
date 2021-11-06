from abc import ABCMeta, abstractmethod
import logging
from typing import Dict, List, Union, Tuple
# from tqdm import trange
from tqdm.auto import trange

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import RobertaTokenizer, DistilBertTokenizer, DistilBertModel
from mdr.retrieval.models.retriever import RobertaCtxEncoder

from utils.model_utils import get_device
from utils.tensor_utils import pad_tensors, to_device

logger = logging.getLogger(__name__)

# Define type aliases
TextPair = Tuple[str, str]


class DenseEncoder(metaclass=ABCMeta):
    @abstractmethod
    def encode_queries(self, queries: List, batch_size: int = None, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = None, **kwargs) -> np.ndarray:
        pass


class MDREncoder(DenseEncoder):
    def __init__(self, model: Union[RobertaCtxEncoder, nn.DataParallel], tokenizer: RobertaTokenizer,
                 max_q_len: int = None, max_p_len: int = None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len

    def encode(self, texts_or_text_pairs: Union[List[str], List[TextPair]],
               max_seq_len: int, batch_size: int = None) -> torch.Tensor:
        total = len(texts_or_text_pairs)
        if batch_size is None or batch_size <= 0:
            batch_size = max(total, 256)
        device = get_device(self.model)
        vectors = []
        self.model.eval()
        with torch.no_grad():
            for batch_start in trange(0, total, batch_size, disable=(total / batch_size <= 10.)):
                inputs = self.tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=texts_or_text_pairs[batch_start:batch_start + batch_size],
                    padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt"
                )
                inputs = to_device({"input_ids": inputs["input_ids"], "input_mask": inputs["attention_mask"]}, device)
                embeddings = self.model(inputs)['embed']
                vectors.append(embeddings.cpu())
        vectors = torch.cat(vectors, dim=0).contiguous()
        assert vectors.shape[0] == total
        return vectors

    def encode_queries(self, queries: List[TextPair], batch_size: int = None, **kwargs) -> np.ndarray:
        try:
            max_q_len = int(kwargs['max_q_len'])
            assert max_q_len > 0
        except:
            max_q_len = self.max_q_len
        return self.encode(queries, max_q_len, batch_size).numpy()

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = None, **kwargs) -> np.ndarray:
        text_pairs = [(para['title'], para['text'] if para['text'] else para['title']) for para in corpus]
        return self.encode(text_pairs, self.max_p_len, batch_size).numpy()


class TASEncoder(DenseEncoder):
    def __init__(self, model: Union[DistilBertModel, nn.DataParallel], tokenizer: DistilBertTokenizer,
                 max_q_len: int = None, max_p_len: int = None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len

    def encode(self, texts: List[str], max_seq_len: int, batch_size: int = None) -> torch.Tensor:
        total = len(texts)
        if batch_size is None or batch_size <= 0:
            batch_size = max(total, 256)
        device = get_device(self.model)
        vectors = []
        self.model.eval()
        with torch.no_grad():
            for batch_start in trange(0, total, batch_size, disable=(total / batch_size <= 10.)):
                inputs = self.tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=texts[batch_start:batch_start + batch_size],
                    padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt"
                )
                inputs = to_device(inputs, device)
                embeddings = self.model(**inputs)[0][:, 0, :]
                vectors.append(embeddings.cpu())
        vectors = torch.cat(vectors, dim=0).contiguous()
        assert vectors.shape[0] == total
        return vectors

    def encode_queries(self, queries: List[str], batch_size: int = None, **kwargs) -> np.ndarray:
        return self.encode(queries, self.max_q_len, batch_size).numpy()

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = None, **kwargs) -> np.ndarray:
        texts = [para['text'] for para in corpus]
        return self.encode(texts, self.max_p_len, batch_size).numpy()


class PassageDataset(Dataset):

    def __init__(self, corpus, tokenizer, max_seq_len):
        super().__init__()
        self.corpus = [(p_id, corpus[p_id]) for p_id in sorted(corpus.keys())]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        p_id, para = self.corpus[index]
        para_codes = self.tokenizer.encode_plus(para["text"], truncation=True, max_length=self.max_seq_len,
                                                return_tensors="pt")
        for k in para_codes.keys():
            para_codes[k].squeeze_(0)
        para_codes['p_id'] = p_id

        return para_codes

    def __len__(self):
        return len(self.corpus)


def collate_passages(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    nn_input = {
        "input_ids": pad_tensors([sample['input_ids'] for sample in samples], pad_id),  # (B, T)
        "attention_mask": pad_tensors([sample['attention_mask'] for sample in samples], 0),  # (B, T)
    }
    if 'token_type_ids' in samples[0]:
        nn_input['token_type_ids'] = pad_tensors([sample['token_type_ids'] for sample in samples], 0),  # (B, T)

    batch = {key: [] for key in samples[0] if key not in nn_input}
    for sample in samples:
        for k in batch:
            batch[k].append(sample[k])
    batch['nn_input'] = nn_input

    return batch
