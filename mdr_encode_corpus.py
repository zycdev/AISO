"""
Description: encode text corpus into a store of dense vectors. 

Usage (adjust the batch size according to your GPU memory):

export MODEL_CHECKPOINT=ckpts/mdr/doc_encoder.pt
export CUDA_VISIBLE_DEVICES=0,1,2,3
python mdr_encode_corpus.py \
  --predict_batch_size 512 \
  --model_name roberta-base \
  --init_checkpoint ${MODEL_CHECKPOINT} \
  --corpus_file data/corpus/hotpot-paragraph-5.tsv \
  --embedding_prefix data/vector/mdr/hotpot-paragraph-5 \
  --max_c_len 300 \
  --num_workers 4 \
  --num_shards 1 \
  --shard_id 0 \
  --strict

"""
import os
import pathlib
import pickle
from tqdm import tqdm
# from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader, Subset

from transformers import AutoConfig, AutoTokenizer
from mdr.retrieval.config import encode_args
from mdr.retrieval.data.encode_datasets import EmDataset, em_collate
from mdr.retrieval.models.retriever import CtxEncoder, RobertaCtxEncoder
from mdr.retrieval.utils.utils import move_to_cuda, load_saved


def main():
    args = encode_args()
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

    bert_config = AutoConfig.from_pretrained(args.model_name)
    if "roberta" in args.model_name:
        model = RobertaCtxEncoder(bert_config, args)
    else:
        model = CtxEncoder(bert_config, args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    eval_dataset = EmDataset(tokenizer, args.corpus_file, args.max_c_len, args.strict)
    shard_size = len(eval_dataset) // args.num_shards
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size if args.shard_id != args.num_shards - 1 else len(eval_dataset)
    sub_eval_dataset = Subset(eval_dataset, list(range(start_idx, end_idx)))
    print(f'Producing encodings for passages [{start_idx:,d}, {end_idx:,d}) '
          f'({args.shard_id}/{args.num_shards} of {len(eval_dataset):,d})')
    eval_dataloader = DataLoader(sub_eval_dataset, batch_size=args.predict_batch_size, collate_fn=em_collate,
                                 pin_memory=True, num_workers=args.num_workers)

    assert args.init_checkpoint != ""
    print(f'Loading the model checkpoint from {args.init_checkpoint} ...')
    model = load_saved(model, args.init_checkpoint, exact=False)
    model.to(device)
    model.eval()

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    embeddings = predict(model, eval_dataloader)
    assert len(embeddings) == end_idx - start_idx
    assert embeddings[0][0] == eval_dataset[start_idx]['p_id']
    assert embeddings[-1][0] == eval_dataset[end_idx - 1]['p_id']

    if args.strict:
        if 'strict' not in args.embedding_prefix:
            args.embedding_prefix = f"{args.embedding_prefix}-strict"
    out_file = f"{args.embedding_prefix}_{args.shard_id}.pkl"
    pathlib.Path(os.path.dirname(out_file)).mkdir(parents=True, exist_ok=True)
    print(f'Encoded passages into {len(embeddings)} x {embeddings[0][1].shape} embeddings, writing to {out_file} ...')
    with open(out_file, mode='wb') as f:
        pickle.dump(embeddings, f)


def predict(model, eval_dataloader):
    model.eval()

    embeddings = []
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch)
        with torch.no_grad():
            out = model(batch_to_feed)
        batch_embedding = out['embed'].cpu().numpy()

        assert len(batch['p_id']) == batch_embedding.shape[0]
        embeddings.extend([(batch['p_id'][i], batch_embedding[i]) for i in range(batch_embedding.shape[0])])

    model.train()
    return embeddings


if __name__ == "__main__":
    main()
