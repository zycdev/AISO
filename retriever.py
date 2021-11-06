import json
import logging
import re
import time
from typing import Dict, List, Tuple

from elasticsearch import Elasticsearch

import numpy as np

from dense_encoder import DenseEncoder
from dense_indexer import DenseIndexer
from utils.utils import chunk

logger = logging.getLogger(__name__)

core_title_pattern = re.compile(r'([^()]+[^\s()])(?:\s*\(.+\))?')


def filter_core_title(x):
    return core_title_pattern.match(x).group(1) if core_title_pattern.match(x) else x


class SparseRetriever(object):
    def __init__(self, index_name='enwiki-20171001-paragraph-5', hosts=('10.208.57.33:9201',),
                 max_retries=4, timeout=15, **kwargs):
        self.index_name = index_name
        self.es = Elasticsearch(hosts, max_retries=max_retries, timeout=timeout, retry_on_timeout=True, **kwargs)

    def pack_query(self, query: str, fields: List = None,
                   must_not: Dict = None, filter_dic: Dict = None, offset: int = 0, size: int = 50) -> Dict:
        if fields is None:
            if 'enwiki' in self.index_name:
                fields = ["title^1.25", "title_unescaped^1.25", "text",
                          "title.bigram^1.25", "title_unescaped.bigram^1.25", "text.bigram"]
            else:
                fields = ["text", "text.bigram"]
        dsl = {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": fields
                        }
                    }
                }
            },
            "from": offset,
            "size": size
        }
        if must_not is not None:
            dsl['query']['bool']['must_not'] = must_not
        if filter_dic:
            dsl['query']['bool']['filter'] = filter_dic  # {"term": {"for_hotpot": True}}
        return dsl

    def search(self, query: str, n_rerank: int = 10, fields: List = None,
               must_not: Dict = None, filter_dic: Dict = None, n_retrieval: int = 50, **kwargs) -> List[Dict]:
        n_retrieval = max(n_rerank, n_retrieval)
        dsl = self.pack_query(query, fields, must_not, filter_dic, size=n_retrieval)
        hits = [hit for hit in self.es.search(dsl, self.index_name, **kwargs)['hits']['hits']]
        if n_rerank > 0:
            hits = self.rerank_with_query(query, hits)[:n_rerank]

        return hits

    def msearch(self, queries: List[str], n_rerank: int = -1, fields: List = None,
                must_not: Dict = None, filter_dic: Dict = None, n_retrieval: int = 50,
                batch_size: int = 64, **kwargs) -> List[List[Dict]]:
        if len(queries) == 0:
            return []

        n_retrieval = max(n_rerank, n_retrieval)
        if not 0 < batch_size <= 64:
            batch_size = 64

        hits_list = []
        for batch_queries in chunk(queries, batch_size):
            body = ["{}\n" + json.dumps(self.pack_query(q, fields, must_not, filter_dic, size=n_retrieval))
                    for q in batch_queries]
            responses = self.es.msearch('\n'.join(body), self.index_name, **kwargs)['responses']
            hits_list.extend([r['hits']['hits'] for r in responses])
        assert len(hits_list) == len(queries)

        if n_rerank > 0:
            hits_list = [self.rerank_with_query(query, hits)[:n_rerank] for query, hits in zip(queries, hits_list)]

        return hits_list

    @staticmethod
    def rerank_with_query(query: str, hits: List[Dict]):
        def score_boost(hit: Dict, q: str):
            title = hit['_source']['title_unescaped']
            core_title = filter_core_title(title)
            q1 = q[4:] if q.startswith('The ') or q.startswith('the ') else q

            score = hit['_score']
            if title in [q, q1]:
                score *= 1.5
            elif title.lower() in [q.lower(), q1.lower()]:
                score *= 1.2
            elif title.lower() in q:
                score *= 1.1
            elif core_title in [q, q1]:
                score *= 1.2
            elif core_title.lower() in [q.lower(), q1.lower()]:
                score *= 1.1
            elif core_title.lower() in q.lower():
                score *= 1.05
            hit['_score'] = score

            return hit

        return sorted([score_boost(hit, query) for hit in hits], key=lambda hit: -hit['_score'])

    @staticmethod
    def format_results(q_ids: List[str], hits_list: List[List[Dict]]) -> Dict[str, Dict[str, float]]:
        results = dict()
        for q_id, hits in zip(q_ids, hits_list):
            results[q_id] = dict((hit['_id'], hit['_score']) for hit in hits)
        return results


class DenseRetriever(object):
    """Does passage retrieving over the provided index and question encoder"""

    def __init__(self, dense_indexer: DenseIndexer, dense_encoder: DenseEncoder):
        self.dense_indexer = dense_indexer
        self.dense_encoder = dense_encoder

    def msearch_(self, vectors: np.ndarray, size: int = 100) -> List[Tuple[List[str], List[float]]]:
        if len(vectors) == 0:
            return []

        t0 = time.time()
        # [(p_ids, scores), ...] shape: (N, 2, size)
        hits_list = self.dense_indexer.search_knn(vectors, size)
        logger.debug(f'dense search time: {time.time() - t0}s')
        return hits_list

    def msearch(self, queries: List, size: int = 100,
                batch_size: int = None, **kwargs) -> List[Tuple[List[str], List[float]]]:
        if len(queries) == 0:
            return []

        vectors = self.dense_encoder.encode_queries(queries, batch_size, **kwargs)  # (N, H)
        hits_list = self.msearch_(vectors, size)  # (N, 2, size)
        return hits_list

    def search(self, query, size: int = 100, **kwargs) -> Tuple[List[str], List[float]]:
        hits = self.msearch([query], size, **kwargs)[0]  # (p_ids, scores), shape: (2, size)
        return hits

    @staticmethod
    def format_results(q_ids: List[str], hits_list: List[Tuple[List[str], List[float]]]) -> Dict[str, Dict[str, float]]:
        results = dict()
        for q_id, hits in zip(q_ids, hits_list):
            results[q_id] = dict((p_id, float(score)) for p_id, score in zip(*hits))
        return results


if __name__ == "__main__":
    sparse_retriever = SparseRetriever('enwiki-20171001-paragraph-5', ['10.208.57.33:9200'], timeout=30)
    print([x['_source']['title'] for x in sparse_retriever.search("In which city did Mark Zuckerberg go to college?")])
    print([[y['_source']['title'] for y in x]
           for x in sparse_retriever.msearch(["In which city did Mark Zuckerberg go to college?"])])
