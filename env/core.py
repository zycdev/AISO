from abc import ABCMeta, abstractmethod
from collections import defaultdict
from html import unescape
import logging
from typing import Union, Optional, Iterable

import faiss

from retriever import SparseRetriever, DenseRetriever

logger = logging.getLogger(__name__)


class BaseEnv(metaclass=ABCMeta):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get(self, p_id: str) -> Optional[dict]:
        pass

    @abstractmethod
    def title2id(self, norm_title: str) -> Optional[str]:
        pass

    @abstractmethod
    def step(self, command: Union[str, tuple], session_id: Optional[str] = None,
             exclusion: Optional[Iterable] = None) -> Optional[str]:
        pass


class Environment(BaseEnv):

    def __init__(self, corpus, title2id, sparse_retriever, dense_retriever, bm25_redis, mdr_redis,
                 for_hotpot=True, strict=False, max_ret_size=500):
        """

        Args:
            corpus (dict):
            title2id (dict):
            sparse_retriever (SparseRetriever):
            dense_retriever (DenseRetriever):
            bm25_redis (redis.Redis):
            mdr_redis (redis.Redis):
            for_hotpot (bool):
            strict (bool):
        """
        self._corpus = corpus
        self._title2id = title2id

        self.query_filter = {"term": {"for_hotpot": True}} if for_hotpot else None
        self.sparse_retriever = sparse_retriever
        self.bm25_redis = bm25_redis
        self.bm25_offset = defaultdict(int)

        self.max_q_len = 70
        self.max_q_sp_len = 350
        self.dense_retriever = dense_retriever
        self.mdr_redis = mdr_redis
        self.mdr_offset = defaultdict(int)

        self.strict = strict
        self.max_ret_size = max_ret_size

    def reset(self):
        self.bm25_offset.clear()
        self.mdr_offset.clear()

    def get(self, p_id):
        if p_id not in self._corpus:
            return None
        para = {"para_id": p_id}
        para.update(self._corpus[p_id])
        para['refs'] = [
            (tuple(span), tgt_title) for tgt_title, anchors in para['hyperlinks'].items() if tgt_title in self._title2id
            for span in anchors
        ]
        return para

    def title2id(self, norm_title):
        return self._title2id.get(norm_title, None)

    def link(self, tgt_title, session_id=None, exclusion=None):
        exclusion = set(exclusion) if exclusion else set()

        if tgt_title not in self._title2id:
            logger.warning(f"{session_id}: invalid link 『{tgt_title}』")
            return None

        tgt_id = self._title2id[tgt_title]
        if tgt_id in exclusion:
            logger.warning(f"{session_id}: link target 『{tgt_title}』 should be excluded")

        return tgt_id

    def bm25(self, query, session_id=None, exclusion=None):
        exclusion = set(exclusion) if exclusion else set()
        query_id = (session_id, query)

        if self.bm25_redis.exists(query) and (self.bm25_redis.llen(query) >= self.max_ret_size or
                                              self.bm25_redis.lindex(query, -1) == 'EOL'):
            hits = self.bm25_redis.lrange(query, 0, -1)
        else:
            hits = [hit['_id'] for hit in self.sparse_retriever.search(query, self.max_ret_size,
                                                                       filter_dic=self.query_filter,
                                                                       n_retrieval=self.max_ret_size * 2)]
            if len(hits) < self.max_ret_size:
                hits.append('EOL')
            self.bm25_redis.delete(query)
            self.bm25_redis.rpush(query, *hits)

        if self.bm25_offset[query_id] >= len(hits):
            return None
        for hit_id in hits[self.bm25_offset[query_id]:]:
            if hit_id == 'EOL':  # don't increase offset if reach the end of retrieval list
                return None
            self.bm25_offset[query_id] += 1
            if hit_id not in exclusion:
                return hit_id
        return None

    def mdr(self, question, expansion=None, session_id=None, exclusion=None):
        if question.endswith('?'):
            question = question[:-1]
        if expansion is None:
            key = question
            query = question
        else:
            sp = self.get(expansion)
            key = f"{question}\t+++\t{unescape(sp['title'])}"
            expansion_text = sp['text']
            if self.strict:
                expansion_text = expansion_text[sp['sentence_spans'][0][0]:sp['sentence_spans'][-1][1]]
            query = (question, expansion_text if expansion_text else sp['title'])
        exclusion = set(exclusion) if exclusion else set()
        query_id = (session_id, key)

        if self.mdr_redis.exists(key) and (self.mdr_redis.llen(key) >= self.max_ret_size or
                                           self.mdr_redis.lindex(key, -1) == 'EOL'):
            hits = self.mdr_redis.lrange(key, 0, -1)
        else:
            faiss.omp_set_num_threads(1)
            hits = self.dense_retriever.search(query, max(self.max_ret_size, 1000),
                                               max_q_len=self.max_q_len if expansion is None else self.max_q_sp_len)[0]
            if len(hits) < max(self.max_ret_size, 1000):
                hits.append('EOL')
            self.mdr_redis.delete(key)
            self.mdr_redis.rpush(key, *hits)

        if self.mdr_offset[query_id] >= len(hits):
            return None
        for hit_id in hits[self.mdr_offset[query_id]:]:
            if hit_id == 'EOL':  # don't increase offset if reach the end of retrieval list
                return None
            self.mdr_offset[query_id] += 1
            if hit_id not in exclusion:
                return hit_id
        return None

    def step(self, command, session_id=None, exclusion=None):
        func_name, argument = command
        if func_name == 'BM25':
            query = argument
            return self.bm25(query, session_id=session_id, exclusion=exclusion)
        elif func_name == 'MDR':
            question, expansion = argument
            return self.mdr(question, expansion, session_id=session_id, exclusion=exclusion)
        elif func_name == 'LINK':
            tgt_title = argument
            return self.link(tgt_title, session_id=session_id, exclusion=exclusion)
        else:
            logger.warning(f'unresolved function: {func_name} in environment')
            return None
