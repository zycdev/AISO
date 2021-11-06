"""
Description: encode text corpus into a store of dense vectors. 

Usage (adjust the batch size according to your GPU memory):

export MODEL_CHECKPOINT=ckpts/mdr/doc_encoder.pt
export DATA_VERSION=5
export CUDA_VISIBLE_DEVICES=3,2,1,0
python encode_corpus.py \
  --model_name roberta-base \
  --init_checkpoint ${MODEL_CHECKPOINT} \
  --corpus_file data/corpus/hotpot-paragraph-${DATA_VERSION}.tsv \
  --embedding_prefix data/vector/mdr/hotpot-paragraph-${DATA_VERSION} \
  --max_c_len 300 \
  --predict_batch_size 512 \
  --num_shards 1 \
  --shard_id 0 \
  --strict

export DATA_VERSION=0
export CUDA_VISIBLE_DEVICES=3,2,1,0
python encode_corpus.py \
  --model_name ckpts/distilbert-dot-tas_b-b256-msmarco \
  --init_checkpoint ckpts/distilbert-dot-tas_b-b256-msmarco \
  --corpus_file data/MSMARCO/passages-${DATA_VERSION}.tsv \
  --embedding_prefix data/vector/tas/msmarco-passage-${DATA_VERSION} \
  --max_c_len 200 \
  --predict_batch_size 512 \
  --num_shards 2 \
  --shard_id 0
"""
import argparse
import logging
import os
import pathlib
import pickle

import torch

from transformers import AutoConfig, AutoModel, AutoTokenizer
from mdr.retrieval.models.retriever import RobertaCtxEncoder

from dense_encoder import MDREncoder, TASEncoder
from utils.data_utils import load_corpus
from utils.model_utils import load_state

logger = logging.getLogger()
logging.basicConfig(format='[%(asctime)s %(levelname)s %(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def main():
    if args.fp16:
        import apex
        apex.amp.register_half_function(torch, 'einsum')

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if "roberta" in args.model_name:
        model_config = AutoConfig.from_pretrained(args.model_name)
        model = RobertaCtxEncoder(model_config, args)
        assert args.init_checkpoint != ""
        logger.info(f'Loading the model checkpoint from {args.init_checkpoint} ...')
        model = load_state(model, args.init_checkpoint, exact=False)
        dense_encoder = MDREncoder(model, tokenizer, max_p_len=args.max_c_len)
    else:
        model = AutoModel.from_pretrained(args.model_name)
        dense_encoder = TASEncoder(model, tokenizer, max_p_len=args.max_c_len)
    dense_encoder.model.to(device)
    dense_encoder.model.eval()
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        dense_encoder.model = amp.initialize(dense_encoder.model, opt_level=args.fp16_opt_level)
    if args.local_rank != -1:
        dense_encoder.model = torch.nn.parallel.DistributedDataParallel(
            dense_encoder.model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    elif n_gpu > 1:
        dense_encoder.model = torch.nn.DataParallel(dense_encoder.model)

    corpus, _ = load_corpus(args.corpus_file)
    if args.strict:
        for p_id, para in corpus.items():
            corpus[p_id]['text'] = para['text'][para['sentence_spans'][0][0]:para['sentence_spans'][-1][1]]
    p_ids = sorted(corpus.keys())
    paras = [corpus[p_id] for p_id in p_ids]

    shard_size = len(paras) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size if args.shard_id != args.num_shards - 1 else len(paras)
    _p_ids = p_ids[start_idx:end_idx]
    _paras = paras[start_idx:end_idx]
    logger.info(f'Producing encodings for passages [{start_idx:,d}, {end_idx:,d}) '
                f'({args.shard_id}/{args.num_shards} of {len(paras):,d})')

    _vectors = dense_encoder.encode_corpus(_paras, args.predict_batch_size)
    _embeddings = [(p_id, _vectors[i]) for i, p_id in enumerate(_p_ids)]
    assert len(_vectors) == end_idx - start_idx
    assert _embeddings[0][0] == p_ids[start_idx] and _embeddings[-1][0] == p_ids[end_idx - 1]

    if args.strict and 'strict' not in args.embedding_prefix:
        args.embedding_prefix = f"{args.embedding_prefix}-strict"
    out_file = f"{args.embedding_prefix}_{args.shard_id}.pkl"
    pathlib.Path(os.path.dirname(out_file)).mkdir(parents=True, exist_ok=True)
    logger.info(f'Encoded {len(_paras)} passages into {len(_embeddings)} x {_embeddings[0][1].shape} embeddings, '
                f'writing to {out_file} ...')
    with open(out_file, mode='wb') as f:
        pickle.dump(_embeddings, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="bert-base-uncased", type=str)
    parser.add_argument("--init_checkpoint", default="", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")

    parser.add_argument('--corpus_file', required=True, type=str, default=None,
                        help='Path to passages .tsv file')
    parser.add_argument("--strict", action="store_true",
                        help="whether to strictly use original data of dataset")
    parser.add_argument("--max_c_len", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--predict_batch_size", default=512, type=int, help="Batch size for prediction")

    parser.add_argument('--embedding_prefix', required=True, type=str, default=None,
                        help='Output path(prefix) to write embeddings to')

    parser.add_argument('--num_shards', type=int, default=1,
                        help="Total amount of data shards")
    parser.add_argument('--shard_id', type=int, default=0,
                        help="Number(0-based) of data shard to process")

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")

    args = parser.parse_args()
    main()
