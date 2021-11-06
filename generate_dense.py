#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.

export CUDA_VISIBLE_DEVICES=0,1,2,3

python generate_dense.py \
  --model_file ckpts/dpr/retriever/multiset/bert-base-encoder.cp \
  --ctx_file data/corpus/enwiki-20171001-paragraph-5.tsv \
  --out_file data/vector/dpr/enwiki-20171001-paragraph-5 \
  --batch_size 256 \
  --num_shards 8 \
  --shard_id 0

"""
import os
import pathlib

import argparse
import logging
import pickle
from tqdm import trange
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from dpr.models import init_biencoder_components
from dpr.options import (add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state,
                         add_tokenizer_params, add_cuda_params)
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint, move_to_device

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def gen_ctx_vectors(ctx_rows: List[Tuple[object, str, str]],
                    model: nn.Module,
                    tensorizer: Tensorizer,
                    insert_title: bool = True) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = args.batch_size
    results = []
    for batch_start in trange(0, n, bsz):
        ctx_ids = []
        batch_token_tensors = []
        for r in ctx_rows[batch_start:batch_start + bsz]:
            ctx_ids.append(r[0])
            batch_token_tensors.append(tensorizer.text_to_tensor(r[1], title=r[2] if insert_title else None))

        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), args.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), args.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), args.device)
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu()

        assert len(ctx_ids) == out.size(0)

        results.extend([(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))])

    return results


def main():
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    print_args(args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.ctx_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank, args.fp16, args.fp16_opt_level)
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')
    logger.debug('saved model keys =%s', saved_state.model_dict.keys())

    prefix_len = len('ctx_model.')
    ctx_state = {key[prefix_len:]: value
                 for (key, value) in saved_state.model_dict.items() if key.startswith('ctx_model.')}
    model_to_load.load_state_dict(ctx_state)

    logger.info(f'reading data from {args.ctx_file} ...')
    rows = []
    with open(args.ctx_file) as tsv_file:
        # file format: doc_id, doc_text, title(, xx)*
        num_field = None
        for line in tsv_file:
            segs = line.strip().split('\t')
            pid, text, title = segs[:3]
            if pid != 'id':
                rows.append((pid, text, title))
            else:
                num_field = len(segs)
            if len(segs) != num_field:
                logger.warning(f'Wrong line format: {pid}')

    shard_size = len(rows) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size if args.shard_id != args.num_shards - 1 else len(rows)
    logger.info(f'Producing encodings for passages [{start_idx:,d}, {end_idx:,d}) '
                f'({args.shard_id}/{args.num_shards} of {len(rows):,d})')
    rows = rows[start_idx:end_idx]

    data = gen_ctx_vectors(rows, encoder, tensorizer, True)

    file = args.out_file + '_' + str(args.shard_id) + '.pkl'
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info(f'{len(data):,d} passages processed. Writing results to {file}')
    with open(file, mode='wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--ctx_file', type=str, default=None, help='Path to passages set .tsv file')
    parser.add_argument('--out_file', required=True, type=str, default=None,
                        help='Output file path (prefix) to write results to')
    parser.add_argument('--shard_id', type=int, default=0, help="Number(0-based) of data shard to process")
    parser.add_argument('--num_shards', type=int, default=1, help="Total amount of data shards")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)

    main()
