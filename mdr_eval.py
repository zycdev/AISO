"""
Evaluating trained retrieval model.

Usage:
export OMP_WAIT_POLICY=PASSIVE
export OMP_NUM_THREADS=16

gpu79
OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=16 python mdr_eval.py \
    --embedded_corpus "data/vector/mdr/hotpot-paragraph-q-strict_*.pkl" \
    --corpus_file data/corpus/hotpot-paragraph-5.tsv \
    --qas_file data/hotpot-dev.tsv \
    --model_name roberta-base \
    --model_path ckpts/mdr/q_encoder.pt \
    --out_file out/mdr/hotpot-dev-5-q.jsonl \
    --index_prefix_path data/index/mdr/hotpot-paragraph-q-strict \
    --index_buffer_size 50000 \
    --max_q_len 70 \
    --max_q_sp_len 350 \
    --batch_size 512 \
    --beam_size_1 1 \
    --beam_size_2 1 \
    --top_k 1 \
    --strict \
    --hnsw \
    --gold_hop1 \
    --only_hop1

gpu21
OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=16 python mdr_eval.py \
    --embedded_corpus "data/vector/mdr/hotpot-paragraph-strict_*.pkl" \
    --corpus_file data/corpus/hotpot-paragraph-5.tsv \
    --qas_file data/hotpot-dev.tsv \
    --model_name roberta-base \
    --model_path ckpts/mdr/q_encoder.pt \
    --out_file out/mdr/hotpot-dev-5.jsonl \
    --index_prefix_path data/index/mdr/hotpot-paragraph-strict \
    --index_buffer_size 50000 \
    --max_q_len 70 \
    --max_q_sp_len 350 \
    --batch_size 512 \
    --beam_size_1 1 \
    --beam_size_2 1 \
    --top_k 1 \
    --strict \
    --hnsw \
    --gold_hop1 \
    --only_hop1
"""
import argparse
import collections
import glob
from html import unescape
import json
import logging
import os
import pathlib

import faiss
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from mdr.retrieval.models.retriever import RobertaCtxEncoder
from mdr.retrieval.utils.basic_tokenizer import SimpleTokenizer
from mdr.retrieval.utils.utils import load_saved, para_has_answer

from dense_encoder import MDREncoder
from dense_indexer import DenseHNSWFlatIndexer, DenseFlatIndexer
from retriever import DenseRetriever
from utils.data_utils import load_qas, load_corpus

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

faiss.omp_set_num_threads(16)


def main():
    logger.info("Loading the dense encoder...")
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    query_encoder = RobertaCtxEncoder(bert_config, args)
    query_encoder = load_saved(query_encoder, args.model_path, exact=False)
    device = torch.device('cuda')
    query_encoder.to(device)
    if torch.cuda.device_count() > 1:
        query_encoder = torch.nn.DataParallel(query_encoder)
    query_encoder.eval()
    dense_encoder = MDREncoder(query_encoder, tokenizer, max_p_len=args.max_q_len)

    # Load or build index
    vector_size = bert_config.hidden_size
    if args.hnsw:
        dense_indexer = DenseHNSWFlatIndexer(vector_size, args.index_buffer_size)
    else:
        dense_indexer = DenseFlatIndexer(vector_size, args.index_buffer_size)
    if args.index_prefix_path and args.hnsw:
        args.index_prefix_path += '.hnsw'
    if (args.index_prefix_path and os.path.exists(args.index_prefix_path + ".index.dpr") and
            os.path.exists(args.index_prefix_path + ".index_meta.dpr")):
        dense_indexer.deserialize_from(args.index_prefix_path)
    else:
        embedding_files = sorted(glob.glob(args.embedded_corpus))
        logger.info(f'Indexing all passage vectors from {", ".join(embedding_files)}...')
        dense_indexer.index_data(embedding_files)
        if args.index_prefix_path:
            pathlib.Path(os.path.dirname(args.index_prefix_path)).mkdir(parents=True, exist_ok=True)
            dense_indexer.serialize(args.index_prefix_path)
        if args.gpu and not args.hnsw:
            res = faiss.StandardGpuResources()
            dense_indexer = faiss.index_cpu_to_gpu(res, 1, dense_indexer)
    retriever = DenseRetriever(dense_indexer, dense_encoder)

    logger.info(f"Loading corpus...")
    corpus, title2id = load_corpus(args.corpus_file, for_hotpot=True, require_hyperlinks=False)

    logger.info(f"Loading qas samples form {args.qas_file} ...")
    qas_samples = load_qas(args.qas_file)
    if args.only_eval_ans:
        qas_samples = [sample for sample in qas_samples if sample[1][1] not in ["yes", "no"]]
    num_qas = len(qas_samples)
    questions = [sample[1][0] for sample in qas_samples]  # (N,)
    questions = [q[:-1] if q.endswith('?') else q for q in questions]

    if args.gold_hop1:
        logger.info("Constructing oracle 2-hop queries ...")
        args.beam_size_1 = 2
        expanded_queries = []  # (N * B1,)
        p_ids1 = []  # (N, B1)
        scores1 = []  # (N, B1)
        for q_idx, (q_id, qas) in enumerate(qas_samples):
            sp_facts = qas[2]
            sp_ids = [title2id[unescape(sp_title)] for sp_title in sp_facts.keys()]
            assert len(sp_ids) == 2
            for sp_id in sp_ids:
                expansion = corpus[sp_id]['text']
                if args.strict:
                    expansion = expansion[corpus[sp_id]['sentence_spans'][0][0]:corpus[sp_id]['sentence_spans'][-1][1]]
                if "roberta" in args.model_name and expansion.strip() == '':
                    expansion = corpus[sp_id]['title']
                expanded_queries.append((questions[q_idx], expansion))
            p_ids1.append(sp_ids)
            scores1.append([0.] * len(sp_ids))
    else:
        logger.info("1-hop searching...")
        hits_list1 = retriever.msearch(questions, args.beam_size_1, args.batch_size)  # (N, 2, B1)
        logger.info("Expanding queries...")
        expanded_queries = []  # (N * B1,)
        p_ids1 = []  # (N, B1)
        scores1 = []  # (N, B1)
        for q_idx, hits in enumerate(hits_list1):
            for hit_idx, (p_id, score) in enumerate(zip(*hits)):
                expansion = corpus[p_id]['text']
                if args.strict:
                    expansion = expansion[corpus[p_id]['sentence_spans'][0][0]:corpus[p_id]['sentence_spans'][-1][1]]
                if "roberta" in args.model_name and expansion.strip() == '':
                    expansion = corpus[p_id]['title']
                    hits_list1[q_idx][1][hit_idx] = float('-inf')
                expanded_queries.append((questions[q_idx], expansion))
            # print(expanded_queries)
            p_ids1.append(hits_list1[q_idx][0])
            scores1.append(hits_list1[q_idx][1])
    p_ids1 = np.array(p_ids1)  # (N, B1)
    scores1 = np.array(scores1)  # (N, B1)
    assert len(expanded_queries) == num_qas * args.beam_size_1 == p_ids1.size == scores1.size

    if args.only_hop1:
        metrics = []
        for q_idx, (q_id, qas) in enumerate(qas_samples):
            question, answer, sp_facts = qas
            p_titles1 = [corpus[p_id]['title'] for p_id in p_ids1[q_idx]]
            sp_titles = set(sp_facts.keys())
            assert len(sp_titles) == 2

            hop1_covered = [sp_title in p_titles1 for sp_title in sp_titles]
            hop1_hit = 1. if np.sum(hop1_covered) > 0 else 0.
            hop1_recall = np.sum(hop1_covered) / len(sp_titles)
            hop1_em = 1. if np.sum(hop1_covered) == len(sp_titles) else 0.

            metrics.append({
                'hop1_hit': hop1_hit,
                "hop1_recall": hop1_recall,
                "hop1_em": hop1_em,
            })
        logger.info(f'hop1-hit: {np.mean([m["hop1_hit"] for m in metrics])}')
        logger.info(f'hop1-Rec.: {np.mean([m["hop1_recall"] for m in metrics])}')
        logger.info(f'hop1-EM: {np.mean([m["hop1_em"] for m in metrics])}')
        return

    logger.info("2-hop searching...")
    # (N * B1, 2, B2)
    hits_list2 = retriever.msearch(expanded_queries, args.beam_size_2, args.batch_size, max_q_len=args.max_q_sp_len)

    # Aggregate path scores
    p_ids2 = []  # (N * B1, B2)
    scores2 = []  # (N * B1, B2)
    for q_idx, hits in enumerate(hits_list2):
        p_ids2.append(hits[0])
        scores2.append(hits[1])
    p_ids2 = np.array(p_ids2).reshape((num_qas, args.beam_size_1, args.beam_size_2))  # (N, B1, B2)
    scores2 = np.array(scores2).reshape((num_qas, args.beam_size_1, args.beam_size_2))  # (N, B1, B2)
    path_scores = np.expand_dims(scores1, axis=2) + scores2  # (N, B1, B2)
    if args.hnsw:
        path_scores = -path_scores

    metrics = []
    retrieval_outputs = []
    simple_tokenizer = SimpleTokenizer()
    for q_idx, (q_id, qas) in enumerate(qas_samples):
        question, answer, sp_facts = qas

        search_scores = path_scores[q_idx]  # (B1, B2)
        ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1],
                                                  (args.beam_size_1, args.beam_size_2))).transpose()
        topk_hop1_titles = []  # (K,)
        topk_2hop_titles = []  # (2K,)
        topk_id_path, topk_title_path = [], []  # (K, 2)
        for rank in range(args.top_k):
            path_indices = ranked_pairs[rank]
            hop_1_id = p_ids1[q_idx, path_indices[0]]
            hop_2_id = p_ids2[q_idx, path_indices[0], path_indices[1]]
            hop_1_title = corpus[hop_1_id]['title']
            hop_2_title = corpus[hop_2_id]['title']
            # assert hop_1_id != hop_2_id
            topk_id_path.append([hop_1_id, hop_2_id])
            topk_title_path.append([hop_1_title, hop_2_title])
            topk_hop1_titles.append(hop_1_title)
            topk_2hop_titles.append(hop_1_title)
            topk_2hop_titles.append(hop_2_title)

        if args.only_eval_ans:
            concat_p = "yes no "
            for id_path in topk_id_path:
                concat_p += " ".join([
                    f"{corpus[p_id]['title']} {corpus[p_id]['text'][corpus[p_id]['sentence_spans'][0][0]:corpus[p_id]['sentence_spans'][-1][1]] if args.strict else corpus[p_id]['text']}"
                    for p_id in id_path
                ])
            metrics.append({
                "question": question,
                "ans_recall": int(para_has_answer([answer], concat_p, simple_tokenizer)),
                "type": "TODO"
            })
        else:
            sp_titles = set(sp_facts.keys())
            assert len(sp_titles) == 2

            sp_covered = [sp_title in topk_2hop_titles for sp_title in sp_titles]
            hop2_hit = 1. if np.sum(sp_covered) > 0 else 0.
            hop2_recall = np.sum(sp_covered) / len(sp_titles)
            hop2_em = 1. if np.sum(sp_covered) == len(sp_titles) else 0.

            hop1_covered = [sp_title in topk_hop1_titles for sp_title in sp_titles]
            hop1_hit = 1. if np.sum(hop1_covered) > 0 else 0.
            hop1_recall = np.sum(hop1_covered) / len(sp_titles)
            hop1_em = 1. if np.sum(hop1_covered) == len(sp_titles) else 0.

            path_covered = [set(title_path) == sp_titles for title_path in topk_title_path]
            path_hits = np.sum(path_covered)
            path_hit = float(path_hits > 0)

            metrics.append({
                "question": question,
                "hop2_hit": hop2_hit,
                "hop2_recall": hop2_recall,
                "hop2_em": hop2_em,
                'hop1_hit': hop1_hit,
                "hop1_recall": hop1_recall,
                "hop1_em": hop1_em,
                "path_hits": path_hits,
                "path_hit": path_hit,
                "type": "TODO"
            })

            # saving when there's no annotations
            candidate_chains = []
            for id_path in topk_id_path:
                candidate_chains.append([corpus[id_path[0]], corpus[id_path[1]]])
            retrieval_outputs.append({
                "_id": q_id,
                "question": question,
                "candidate_chains": candidate_chains,
                # "sp": sp_chain,
                # "answer": gold_answers,
                # "type": type_,
                # "covered_k": covered_k
            })

    if args.out_file:
        with open(args.out_file, "w") as out:
            for item in retrieval_outputs:
                out.write(json.dumps(item) + "\n")

    logger.info(f"Evaluating {len(metrics)} samples...")
    type2items = collections.defaultdict(list)
    for item in metrics:
        type2items[item["type"]].append(item)
    if args.only_eval_ans:
        logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in metrics])}')
        # for t in type2items.keys():
        #     logger.info(f"{t} Questions num: {len(type2items[t])}")
        #     logger.info(f'Ans Recall: {np.mean([m["ans_recall"] for m in type2items[t]])}')
    else:
        logger.info(f'2hop-hit: {np.mean([m["hop2_hit"] for m in metrics])}')
        logger.info(f'2hop-Rec.: {np.mean([m["hop2_recall"] for m in metrics])}')
        logger.info(f'2hop-EM: {np.mean([m["hop2_em"] for m in metrics])}')
        logger.info(f'hop1-hit: {np.mean([m["hop1_hit"] for m in metrics])}')
        logger.info(f'hop1-Rec.: {np.mean([m["hop1_recall"] for m in metrics])}')
        logger.info(f'hop1-EM: {np.mean([m["hop1_em"] for m in metrics])}')
        logger.info(f'path-hits: {np.mean([m["path_hits"] for m in metrics])}')
        logger.info(f'path-hit: {np.mean([m["path_hit"] for m in metrics])}')
        # for t in type2items.keys():
        #     logger.info(f"{t} Questions num: {len(type2items[t])}")
        #     logger.info(f'\tAvg SP-hit: {np.mean([m["hop2_hit"] for m in type2items[t]])}')
        #     logger.info(f'\tAvg SP-EM: {np.mean([m["hop2_em"] for m in type2items[t]])}')
        #     logger.info(f'Avg Hop1-hit: {np.mean([m["hop1_hit"] for m in type2items[t]])}')
        #     # logger.info(f'Path Recall: {np.mean([m["path_covered"] for m in type2items[t]])}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qas_file', required=True, type=str, default=None,
                        help="TSV file of which each line contains question, answer and support facts "
                             "with format 'question \\t answer \\t sp'")
    parser.add_argument('--embedded_corpus', required=True, type=str, default=None,
                        help='Glob path to encoded passages')
    parser.add_argument('--corpus_file', required=True, type=str, default=None,
                        help="TSV file that contains all passages with format: 'id \\t text \\t title ...'")

    parser.add_argument("--strict", action="store_true", help="whether to strictly use original data of dataset")

    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--model_path', required=True, type=str, default=None)

    parser.add_argument("--index_prefix_path", type=str, default=None,
                        help='Index file prefix path to load/save')
    parser.add_argument('--index_buffer_size', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument('--hnsw', action="store_true")
    parser.add_argument('--gpu', action="store_true", help="Whether search on gpu")

    parser.add_argument('--top_k', type=int, default=2, help="top-k paths")
    parser.add_argument('--beam_size_1', type=int, default=5)
    parser.add_argument('--beam_size_2', type=int, default=5)
    parser.add_argument('--max_q_len', type=int, default=70)
    parser.add_argument('--max_q_sp_len', type=int, default=350)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--gold_hop1', action="store_true")
    parser.add_argument('--only_hop1', action="store_true")
    parser.add_argument('--only_eval_ans', action="store_true")

    parser.add_argument("--out_file", type=str, default=None, help='Output .jsonl file path to save retrieval results')

    args = parser.parse_args()
    main()
