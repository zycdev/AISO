from collections import defaultdict
from html import unescape
import logging

import faiss

from retriever import SparseRetriever, DenseRetriever

logger = logging.getLogger(__name__)


class WikiWorld(object):
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
        self.title2id = title2id

        self.query_filter = {"term": {"for_hotpot": True}} if for_hotpot else None
        self.sparse_retriever = sparse_retriever
        # redis.Redis(host='10.60.1.79', port=6379, db=0, password='redis4zyc', decode_responses=True)
        self.bm25_redis = bm25_redis
        self.bm25_offset = defaultdict(int)

        self.max_q_len = 70
        self.max_q_sp_len = 350
        self.dense_retriever = dense_retriever
        # redis.Redis(host='10.60.1.79', port=6379, db=1, password='redis4zyc', decode_responses=True)
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
        para['refs'] = {para['text'][span[0]:span[1]]: (tgt_title, span)
                        for tgt_title, anchors in para['hyperlinks'].items() for span in anchors}
        return para

    def link(self, tgt_title, q_id=None, excluded=None):
        if excluded is not None:
            excluded = set(excluded)

        if tgt_title not in self.title2id:
            logger.warning(f"{q_id}: invalid link 『{tgt_title}』")
            return None

        tgt_id = self.title2id[tgt_title]
        if excluded is not None and tgt_id in excluded:
            logger.warning(f"{q_id}: link target 『{tgt_title}』 should be excluded")

        return tgt_id

    def bm25(self, query, q_id=None, excluded=None):
        if excluded is not None:
            excluded = set(excluded)
        session_id = (q_id, query)

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

        if self.bm25_offset[session_id] >= len(hits):
            return None
        for hit_id in hits[self.bm25_offset[session_id]:]:
            if hit_id == 'EOL':  # don't increase offset if reach the end of retrieval list
                return None
            self.bm25_offset[session_id] += 1
            if excluded is None or hit_id not in excluded:
                return hit_id
        return None

    def mdr(self, question, expansion=None, q_id=None, excluded=None):
        if question.endswith('?'):
            question = question[:-1]
        if excluded is not None:
            excluded = set(excluded)
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
        session_id = (q_id, key)

        if self.mdr_redis.exists(key) and (self.mdr_redis.llen(key) >= self.max_ret_size or
                                           self.mdr_redis.lindex(key, -1) == 'EOL'):
            hits = self.mdr_redis.lrange(key, 0, -1)
        else:
            faiss.omp_set_num_threads(1)
            hits = self.dense_retriever.search(query, max(self.max_ret_size, 1000),
                                               self.max_q_len if expansion is None else self.max_q_sp_len)[0]
            if len(hits) < max(self.max_ret_size, 1000):
                hits.append('EOL')
            self.mdr_redis.delete(key)
            self.mdr_redis.rpush(key, *hits)

        if self.mdr_offset[session_id] >= len(hits):
            return None
        for hit_id in hits[self.mdr_offset[session_id]:]:
            if hit_id == 'EOL':  # don't increase offset if reach the end of retrieval list
                return None
            self.mdr_offset[session_id] += 1
            if excluded is None or hit_id not in excluded:
                return hit_id
        return None

    def execute(self, command, q_id=None, excluded=None):
        func_name = command[0]
        if func_name == 'BM25':
            query = command[1]
            return self.bm25(query, q_id=q_id, excluded=excluded)
        elif func_name == 'MDR':
            question, expansion = command[1]
            return self.mdr(question, expansion, q_id=q_id, excluded=excluded)
        elif func_name == 'LINK':
            tgt_title = command[1]
            return self.link(tgt_title, q_id=q_id, excluded=excluded)
        else:
            logger.warning(f'unresolved func: {func_name} in WikiWorld')
            return None
