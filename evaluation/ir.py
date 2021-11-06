import logging
from typing import List, Dict, Tuple

import pytrec_eval

logger = logging.getLogger(__name__)


def evaluate_trec(qrels: Dict[str, Dict[str, int]], results: Dict[str, Dict[str, float]],
                  k_values: List[int]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    ndcg, _map, recall, precision = dict(), dict(), dict(), dict()
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    for q_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[q_id][f"ndcg_cut_{k}"]
            _map[f"MAP@{k}"] += scores[q_id][f"map_cut_{k}"]
            recall[f"Recall@{k}"] += scores[q_id][f"recall_{k}"]
            precision[f"P@{k}"] += scores[q_id][f"P_{k}"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(qrels) * 100., 3)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(qrels) * 100., 3)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(qrels) * 100., 3)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(qrels) * 100., 3)

    return ndcg, _map, recall, precision


def evaluate_custom(qrels: Dict[str, Dict[str, int]], results: Dict[str, Dict[str, float]],
                    k_values: List[int], metric: str) -> Tuple[Dict[str, float]]:
    if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
        return mrr(qrels, results, k_values)

    elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
        return recall_cap(qrels, results, k_values)

    elif metric.lower() in ["hole", "hole@k"]:
        return hole(qrels, results, k_values)


def mrr(qrels: Dict[str, Dict[str, int]], results: Dict[str, Dict[str, float]],
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    measures = dict()
    for k in k_values:
        measures[f"MRR@{k}"] = 0.0

    k_max, top_hits = max(k_values), dict()
    for q_id, hits in results.items():
        top_hits[q_id] = sorted(hits.items(), key=lambda item: item[1], reverse=True)[0:k_max]

    for q_id in set(qrels) & set(top_hits):
        q_relevant_paras = set([p_id for p_id in qrels[q_id] if qrels[q_id][p_id] > 0])
        for k in k_values:
            for rank, hit in enumerate(top_hits[q_id][0:k]):
                if hit[0] in q_relevant_paras:
                    measures[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        measures[f"MRR@{k}"] = round(measures[f"MRR@{k}"] / len(qrels) * 100., 3)

    return measures


def recall_cap(qrels: Dict[str, Dict[str, int]], results: Dict[str, Dict[str, float]],
               k_values: List[int]) -> Tuple[Dict[str, float]]:
    measures = dict()
    for k in k_values:
        measures[f"R_cap@{k}"] = 0.0

    k_max = max(k_values)
    for query_id, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        query_relevant_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        for k in k_values:
            retrieved_docs = [row[0] for row in top_hits[0:k] if qrels[query_id].get(row[0], 0) > 0]
            denominator = min(len(query_relevant_docs), k)
            measures[f"R_cap@{k}"] += (len(retrieved_docs) / denominator)

    for k in k_values:
        measures[f"R_cap@{k}"] = round(measures[f"R_cap@{k}"] / len(results) * 100., 3)

    return measures


def hole(qrels: Dict[str, Dict[str, int]], results: Dict[str, Dict[str, float]],
         k_values: List[int]) -> Tuple[Dict[str, float]]:
    measures = {}
    for k in k_values:
        measures[f"Hole@{k}"] = 0.0

    annotated_corpus = set()
    for _, docs in qrels.items():
        for doc_id, score in docs.items():
            annotated_corpus.add(doc_id)

    k_max = max(k_values)
    for _, scores in results.items():
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        for k in k_values:
            hole_docs = [row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus]
            measures[f"Hole@{k}"] += len(hole_docs) / k

    for k in k_values:
        measures[f"Hole@{k}"] = round(measures[f"Hole@{k}"] / len(results) * 100., 3)

    return measures
