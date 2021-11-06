"""
Usage:

CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 train_union.py \
    --corpus_path data/corpus/hotpot-paragraph-4.min.tsv \
    --train_file data/hotpot-step-train.jsonl \
    --predict_file data/hotpot-step-dev.jsonl \
    --do_train \
    --do_predict \
    --encoder_name google/electra-base-discriminator \
    --init_checkpoint "" \
    --hard_negs_per_state 2 \
    --memory_size 2 \
    --max_distractors 2 \
    --num_workers 1 \
    --cmd_dropout_prob 0 \
    --sp_weight 0.05 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 100 \
    --eval_period -1 \
    --criterion_metric action_acc \
    --tag no-early-answer \
    --debug

gpu216 gpu83 gpu79
export TOKENIZERS_PARALLELISM=true
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_union.py \
    --corpus_path data/corpus/hotpot-paragraph-5.strict.refined.tsv \
    --train_file data/hotpot-step-train.strict.refined.jsonl \
    --predict_file data/hotpot-step-dev.strict.refined.jsonl \
    --do_train \
    --do_predict \
    --encoder_name google/electra-base-discriminator \
    --init_checkpoint "" \
    --hard_negs_per_state 2 \
    --memory_size 2 \
    --max_distractors 2 \
    --strict \
    --num_workers 0 \
    --cmd_dropout_prob 0.5 \
    --sp_weight 0.5 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_infer_batch_size 16 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 30 \
    --eval_period 1000 \
    --criterion_metric cmd_acc \
    --tag rand-state-per-quest \
    --comment td5-exp1-ilma.4-cmd15

gpu21
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=13579 train_union.py \
    --corpus_path data/corpus/hotpot-paragraph-5.strict.refined.tsv \
    --train_file data/hotpot-step-train.strict.refined.jsonl \
    --predict_file data/hotpot-step-dev.strict.refined.jsonl \
    --do_train \
    --do_predict \
    --encoder_name google/electra-large-discriminator \
    --init_checkpoint "" \
    --hard_negs_per_state 2 \
    --memory_size 2 \
    --max_distractors 2 \
    --strict \
    --num_workers 0 \
    --cmd_dropout_prob 0.5 \
    --sp_weight 0.5 \
    --gradient_accumulation_steps 2\
    --per_gpu_train_batch_size 8 \
    --per_gpu_infer_batch_size 16 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 30 \
    --eval_period 1000 \
    --criterion_metric cmd_acc \
    --tag td5-exp1-ila.4 \
    --comment pld-sb0-x1-f+a
CUDA_VISIBLE_DEVICES=3,2,0 python -m torch.distributed.launch --nproc_per_node=3 --master_port=24680 train_union.py \
    --corpus_path data/corpus/hotpot-paragraph-5.strict.refined.tsv \
    --train_file data/hotpot-step-train.strict.refined.jsonl \
    --predict_file data/hotpot-step-dev.strict.refined.jsonl \
    --do_train \
    --do_predict \
    --encoder_name google/electra-large-discriminator \
    --init_checkpoint "" \
    --hard_negs_per_state 2 \
    --memory_size 2 \
    --max_distractors 2 \
    --strict \
    --num_workers 0 \
    --cmd_dropout_prob 0.5 \
    --sp_weight 0.5 \
    --gradient_accumulation_steps 2\
    --per_gpu_train_batch_size 8 \
    --per_gpu_infer_batch_size 16 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 30 \
    --eval_period 1000 \
    --criterion_metric cmd_acc \
    --tag td5-exp1-ila.4-woin \
    --comment pld-sb0-wo*-cmd10

CUDA_VISIBLE_DEVICES=0 python train_union.py \
    --corpus_path data/corpus/hotpot-paragraph-5.strict.refined.tsv \
    --train_file data/hotpot-step-train.strict.refined.jsonl \
    --predict_file data/hotpot-step-dev.strict.refined.jsonl \
    --do_train \
    --do_predict \
    --encoder_name google/electra-base-discriminator \
    --init_checkpoint "" \
    --hard_negs_per_state 2 \
    --memory_size 2 \
    --max_distractors 2 \
    --strict \
    --num_workers 0 \
    --cmd_dropout_prob 0.5 \
    --sp_weight 0.5 \
    --gradient_accumulation_steps 1\
    --per_gpu_train_batch_size 32 \
    --per_gpu_infer_batch_size 32 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 30 \
    --eval_period 1000 \
    --criterion_metric cmd_acc \
    --tag td5-exp1-ila.4 \
    --comment pld-sb0-x1


export CKPT_PATH=ckpts/alpha_electra-base-discriminator_DP0.0_HN2_M2_D2_adamW_SP0.05_B24_LR5.0e-05-WU0.1-E100_S42_01150136/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=1 python train_union.py \
    --corpus_path data/corpus/hotpot-paragraph-4.min.tsv \
    --train_file data/hotpot-step-train.jsonl \
    --predict_file data/hotpot-step-dev.jsonl \
    --do_predict \
    --encoder_name google/electra-base-discriminator \
    --init_checkpoint $CKPT_PATH \
    --hard_negs_per_state 2 \
    --memory_size 2 \
    --max_distractors 2 \
    --num_workers 1 \
    --cmd_dropout_prob 0 \
    --sp_weight 0.05 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs 2 \
    --eval_period 100 \
    --criterion_metric action_acc \
    --tag alpha
"""
from datetime import datetime
from functools import partial
import json
import logging
import os
from tqdm import tqdm, trange
# from tqdm.auto import tqdm, trange

import numpy as np
import torch
import torch.multiprocessing
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup  # AutoConfig,

from config import train_args, ADDITIONAL_SPECIAL_TOKENS, FUNCTIONS, NA_POS
from models.union_model import UnionModel
from transition_data import collate_transitions, TransitionDataset, ConstantDataset
from utils.data_utils import load_corpus
from utils.model_utils import load_state, save_model
from utils.tensor_utils import to_cuda
from utils.text_utils import finetune_start
from utils.utils import set_seed, flatten_dict, collection_f1, harmonic_mean
from hotpot_evaluate_plus import exact_match_score, f1_score

logger = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')

# if logger.hasHandlers():
#     logger.handlers.clear()

l2a_from = 0
save_steps = (54, 59, 60, 66, 68, 72, 74, 79)
save_epochs = (22, 23, 24, 25, 26, 27, 28, 29)


# noinspection PyUnboundLocalVariable
def main():
    args = train_args()
    if args.fp16:
        import apex
        apex.amp.register_half_function(torch, 'einsum')

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpu)
    args.infer_batch_size = args.per_gpu_infer_batch_size * max(1, n_gpu)

    if args.strict:
        if 'strict' not in args.corpus_path:
            args.corpus_path = f"{args.corpus_path[:-4]}-strict.tsv"
        if 'strict' not in args.train_file:
            args.train_file = f"{args.train_file[:-6]}-strict.jsonl"
        if 'strict' not in args.predict_file:
            args.predict_file = f"{args.predict_file[:-6]}-strict.jsonl"

    # config logger and experiment dir
    formatter = logging.Formatter('[%(asctime)s %(levelname)s %(name)s:%(lineno)s] %(message)s',
                                  datefmt='%m/%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    if args.do_train:
        tt_batch_size = (args.train_batch_size *
                         args.gradient_accumulation_steps *
                         (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        hyper_params = {
            "spw": args.sp_weight,
            "hn": args.hard_negs_per_state,
            "m": args.memory_size,
            "d": args.max_distractors
        }
        exp_dir = (f"{args.encoder_name.split('/')[-1]}_DP{args.cmd_dropout_prob}_"
                   f"HN{args.hard_negs_per_state}_M{args.memory_size}_D{args.max_distractors}_"
                   f"{'adam' if args.use_adam else 'adamW'}_SP{args.sp_weight}_"
                   f"B{tt_batch_size}_LR{args.learning_rate:.1e}_WU{args.warmup_ratio}_E{args.num_train_epochs}_"
                   f"{f'fp16{args.fp16_opt_level}_' if args.fp16 else ''}"
                   f"S{args.seed}_{datetime.now().strftime('%m%d%H%M')}")
        if args.tag:
            exp_dir = f"{args.tag}_{exp_dir}"
        if args.comment:
            exp_dir = f"{exp_dir}_{args.comment}"
        output_dir = os.path.join(args.output_dir, exp_dir)
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the process 0 mkdir and write args
        else:
            if os.path.exists(output_dir) and os.listdir(output_dir):
                print(f"output directory {output_dir} already exists and is not empty.")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'args.json'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)
        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the process 0 mkdir and write args

        file_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        # noinspection PyArgumentList
        logging.basicConfig(
            format='[%(asctime)s %(levelname)s %(name)s] %(message)s', datefmt='%m/%d %H:%M:%S',
            level=logging.DEBUG if args.local_rank in [-1, 0] else logging.WARNING,
            handlers=[file_handler, stream_handler]
        )
    else:
        # noinspection PyArgumentList
        logging.basicConfig(
            format='[%(asctime)s %(levelname)s %(name)s] %(message)s', datefmt='%m/%d %H:%M:%S',
            level=logging.DEBUG if args.local_rank in [-1, 0] else logging.WARNING,
            handlers=[stream_handler]
        )
    # logger.setLevel(logging.DEBUG if args.local_rank in [-1, 0] else logging.WARNING)
    logger.warning(f"Process rank: {args.local_rank}, device: {device}, n_gpu: {n_gpu}")
    logger.info(args)

    set_seed(args.seed)

    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()  # Make sure only the process 0 will download model & vocab
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, use_fast=True,
                                              additional_special_tokens=list(ADDITIONAL_SPECIAL_TOKENS.values()))
    model = UnionModel(args.encoder_name, args.max_ans_len, args.sp_weight)
    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the process 0 will download model & vocab

    if args.init_checkpoint:
        logger.info(f"Loading model from {args.init_checkpoint}...")
        model = load_state(model, args.init_checkpoint)
    model.to(device)
    logger.info(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}")

    logger.info(f"Loading corpus from {args.corpus_path}...")
    corpus, title2id = load_corpus(args.corpus_path, for_hotpot=True, require_hyperlinks=True)

    collate_func = partial(collate_transitions, pad_id=tokenizer.pad_token_id)
    if args.local_rank in [-1, 0] or args.debug:
        eval_dataset = TransitionDataset(args.predict_file, tokenizer, corpus, title2id,
                                         args.max_seq_len, args.max_q_len, args.max_obs_len,
                                         args.hard_negs_per_state, args.memory_size, args.max_distractors, args.strict)
        eval_dataset_imgs = [ConstantDataset([eval_dataset[i] for i in range(len(eval_dataset))]) for _ in range(2)]
        if args.debug:
            eval_dataset_imgs = [Subset(data_image, indices=list(range(1024))) for data_image in eval_dataset_imgs]
        eval_dataloaders = [
            DataLoader(data_image, batch_size=args.infer_batch_size, collate_fn=collate_func,
                       pin_memory=True, num_workers=args.num_workers) for data_image in eval_dataset_imgs
        ]

    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        if args.use_adam:
            optimizer = Adam(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        else:
            optimizer = AdamW(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        if args.fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    else:
        if args.fp16:
            from apex import amp
            model = amp.initialize(model, opt_level=args.fp16_opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter(os.path.join('runs/', exp_dir))

        train_dataset = TransitionDataset(args.train_file, tokenizer, corpus, title2id,
                                          args.max_seq_len, args.max_q_len, args.max_obs_len,
                                          args.hard_negs_per_state, args.memory_size, args.max_distractors, args.strict)
        if args.debug:
            train_dataset = eval_dataset_imgs[0]
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                      collate_fn=collate_func, pin_memory=True, num_workers=args.num_workers)

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        warmup_steps = t_total * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {tt_batch_size}")
        logger.info(f"  Total optimization steps = {t_total}")
        global_step = 0  # gradient update step
        batch_step = 0  # forward batch count
        best_step, best_criterion, best_metrics = 0, 0, dict()
        best_pr, best_qa, best_joint = 0, 0, 0
        losses, last_losses = None, None
        model.train()
        epoch_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
        for epoch in epoch_iterator:
            batch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for batch in batch_iterator:
                batch_step += 1

                nn_input = batch['nn_input'] if args.no_cuda else to_cuda(batch['nn_input'])
                if epoch < l2a_from:
                    nn_input['action_label'].fill_(-1)
                _losses = model(nn_input)

                if n_gpu > 1:
                    for k in _losses.keys():  # mean() to average on multi-gpu parallel (not distributed) training
                        _losses[k] = _losses[k].mean()
                if args.gradient_accumulation_steps > 1:
                    for k in _losses.keys():
                        _losses[k] = _losses[k] / args.gradient_accumulation_steps
                    # criterion_loss = criterion_loss / args.gradient_accumulation_steps
                criterion_loss = _losses['all']

                if args.fp16:
                    with amp.scale_loss(criterion_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    criterion_loss.backward()

                if losses is None:
                    losses = {k: v.item() for k, v in _losses.items()}
                    last_losses = {k: 0.0 for k, v in _losses.items()}
                else:
                    for k in losses.keys():
                        losses[k] += _losses[k].item()

                if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    # log losses
                    if args.local_rank in [-1, 0] and args.log_period > 0 and global_step % args.log_period == 0:
                        try:
                            tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                        except:
                            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        for k in losses:
                            tb_writer.add_scalar(f'loss/{k}', (losses[k] - last_losses[k]) / args.log_period,
                                                 global_step)
                            last_losses[k] = losses[k]
                        # last_losses = losses.copy()

                    # log metrics
                    if (args.local_rank in [-1, 0] and epoch >= 5 and
                            args.eval_period != -1 and global_step % args.eval_period == 0):
                        metrics = evaluates(args, model, tokenizer, eval_dataloaders, eval_dataset.examples)
                        criterion = metrics[args.criterion_metric]
                        logger.info(f"epoch:{epoch:2d} | step {global_step:5d} | "
                                    f"loss:{losses['all'] / global_step:2.4f} | "
                                    f"{args.criterion_metric} {criterion:2.2f}")
                        for k, v in metrics.items():
                            tb_writer.add_scalar(f"metric/{k}", v, global_step)
                        if best_criterion < criterion:
                            logger.info(f"Saving model with best {args.criterion_metric} "
                                        f"{best_criterion:.2f} -> {criterion:.2f}")
                            save_model(model, os.path.join(output_dir, "checkpoint_best.pt"))
                            best_criterion = criterion
                            best_metrics = metrics
                            best_step = global_step
                        if best_pr < metrics['para_f1']:
                            logger.info(f"Saving model with best para_f1 {best_pr:.2f} -> {metrics['para_f1']:.2f}")
                            save_model(model, os.path.join(output_dir, "checkpoint_pr.pt"))
                            best_pr = metrics['para_f1']
                        if best_qa < metrics['answer_f1']:
                            logger.info(f"Saving model with best answer_f1 {best_qa:.2f} "
                                        f"-> {metrics['answer_f1']:.2f}")
                            save_model(model, os.path.join(output_dir, "checkpoint_qa.pt"))
                            best_qa = metrics['answer_f1']
                        if best_joint < metrics['joint_acc']:
                            logger.info(f"Saving model with best joint_acc {best_joint:.2f} "
                                        f"-> {metrics['joint_acc']:.2f}")
                            save_model(model, os.path.join(output_dir, "checkpoint_joint.pt"))
                            best_joint = metrics['joint_acc']
                        if global_step / 1000 in save_steps:
                            save_model(model, os.path.join(output_dir, f"checkpoint_{global_step}.pt"))

                batch_iterator.set_description('Iteration(loss=%2.4f)' %
                                               (criterion_loss.item() * args.gradient_accumulation_steps))
            batch_iterator.close()

            if args.local_rank in [-1, 0]:
                save_model(model, os.path.join(output_dir, "checkpoint_last.pt"))
                metrics = evaluates(args, model, tokenizer, eval_dataloaders, eval_dataset.examples)
                criterion = metrics[args.criterion_metric]
                logger.info(f"end of epoch {epoch:2d} | loss {losses['all'] / global_step:2.4f} | "
                            f"{args.criterion_metric} {criterion:2.2f}")
                for k, v in metrics.items():
                    tb_writer.add_scalar(f"metric/{k}", v, global_step)
                if best_criterion < criterion:
                    logger.info(f"Saving model with best {args.criterion_metric} "
                                f"{best_criterion:.2f} -> {criterion:.2f}")
                    save_model(model, os.path.join(output_dir, "checkpoint_best.pt"))
                    best_criterion = criterion
                    best_metrics = metrics
                    best_step = global_step
                if best_pr < metrics['para_f1']:
                    logger.info(f"Saving model with best para_f1 {best_pr:.2f} -> {metrics['para_f1']:.2f}")
                    save_model(model, os.path.join(output_dir, "checkpoint_pr.pt"))
                    best_pr = metrics['para_f1']
                if best_qa < metrics['answer_f1']:
                    logger.info(f"Saving model with best answer_f1 {best_qa:.2f} -> {metrics['answer_f1']:.2f}")
                    save_model(model, os.path.join(output_dir, "checkpoint_qa.pt"))
                    best_qa = metrics['answer_f1']
                if best_joint < metrics['joint_acc']:
                    logger.info(f"Saving model with best joint_acc {best_joint:.2f} "
                                f"-> {metrics['joint_acc']:.2f}")
                    save_model(model, os.path.join(output_dir, "checkpoint_joint.pt"))
                    best_joint = metrics['joint_acc']
                if epoch in save_epochs:
                    save_model(model, os.path.join(output_dir, f"checkpoint_{epoch}.pt"))
        epoch_iterator.close()

        if args.local_rank in [-1, 0]:
            tb_writer.add_hparams(hyper_params,
                                  {f"M/{k}": v for k, v in flatten_dict(best_metrics, sep='/').items()})
            tb_writer.close()
            logger.info(f"Achieve {best_criterion:.2f} {args.criterion_metric} at step {best_step}")
            if best_step / 1000 in save_steps:
                os.remove(os.path.join(output_dir, f"checkpoint_{best_step}.pt"))

    elif args.do_predict:
        metrics = evaluates(args, model, tokenizer, eval_dataloaders, eval_dataset.examples)
        logger.info(f"test performance {metrics}")

    elif args.do_test:
        pass


def evaluates(args, model, tokenizer, eval_dataloaders, qas_examples):
    assert len(eval_dataloaders) > 0
    all_metrics = [evaluate(args, model, tokenizer, eval_dataloaders[i], qas_examples)
                   for i in range(len(eval_dataloaders))]
    if len(all_metrics) == 1:
        return all_metrics[0]
    avg_metrics = {}
    for k in all_metrics[0]:
        avg_metrics[k] = np.mean([all_metrics[i][k] for i in range(len(all_metrics))])
    return avg_metrics


def evaluate(args, model, tokenizer, eval_dataloader, qas_examples):
    obs_accuracies = []
    para_accuracies = []
    para_f1_scores = []
    sent_accuracies = []
    sent_f1_scores = []
    action_accuracies = []
    cmd_accuracies = []
    joint_accuracies = []
    link_accuracies = {"all": [], "should": [], "practical": [], "act": []}
    answer_ems = {"all": [], "should": [], "practical": [], "act": []}
    answer_f1s = {"all": [], "should": [], "practical": [], "act": []}
    model.eval()
    for batch in tqdm(eval_dataloader):
        nn_input = batch['nn_input'] if args.no_cuda else to_cuda(batch['nn_input'])
        with torch.no_grad():
            outputs = model(nn_input)
            # (B, _P)    (B, _S)      (B,)            (B,)
            para_logits, sent_logits, para_threshold, sent_threshold = outputs[4:8]
            # (B,)       (B,)        (B,)      (B,)       (B,)
            pred_action, pred_start, pred_end, pred_link, pred_exp = outputs[8:13]

            para_preds = [(_para_logits > _para_threshold).float()  # (B, _P)
                          for _para_logits, _para_threshold in zip(para_logits, para_threshold)]
            sent_preds = [(_sent_logits > _sent_threshold).float()  # (B, _S)
                          for _sent_logits, _sent_threshold in zip(sent_logits, sent_threshold)]

            obs_accuracies.extend(
                [float(_para_preds[0] == _paras_label[0])
                 for _para_preds, _paras_label in zip(para_preds, nn_input['paras_label']) if len(_para_preds) > 0]
            )
            para_accuracies.extend(
                (torch.cat(para_preds) == torch.cat(nn_input['paras_label'])).tolist()
            )
            para_f1_scores.extend(
                [collection_f1(_para_preds.nonzero().squeeze_(1).tolist(), _paras_label.nonzero().squeeze_(1).tolist())
                 for _para_preds, _paras_label in zip(para_preds, nn_input['paras_label']) if len(_para_preds) > 0]
            )
            sent_accuracies.extend(
                (torch.cat(sent_preds) == torch.cat(nn_input['sents_label'])).tolist()
            )
            sent_f1_scores.extend(
                [collection_f1(_sent_preds.nonzero().squeeze_(1).tolist(), _sents_label.nonzero().squeeze_(1).tolist())
                 for _sent_preds, _sents_label in zip(sent_preds, nn_input['sents_label']) if len(_sent_preds) > 0]
            )
            action_accuracies.extend((pred_action == nn_input['action_label']).float().tolist())
            # link_accuracies['all'].extend((pred_link == nn_input['link_label']).float().tolist())

        pred_start = pred_start.tolist()
        pred_end = pred_end.tolist()
        for i, q_id in enumerate(batch['q_id']):
            qas_example = qas_examples[q_id]

            # predict answer
            context_token_offset = nn_input['context_token_offset'][i].item()
            start_token = pred_start[i] - context_token_offset
            end_token = pred_end[i] - context_token_offset
            token_ids = nn_input['input_ids'][i].tolist()
            context_token_ids = token_ids[context_token_offset:]
            context_token_spans = batch['context_token_spans'][i]
            if start_token < 0:
                if pred_start[i] != pred_end[i] or pred_start[i] not in [1, 2, NA_POS]:
                    logger.warning(f"predicted unexpected ans_span [{pred_start[i]}, {pred_end[i]}], "
                                   f"gold: [{nn_input['answer_starts'][i][0]}, {nn_input['answer_ends'][i][0]}]")
                if pred_start[i] == 1:
                    pred_ans = 'yes'
                elif pred_start[i] == 2:
                    pred_ans = 'no'
                else:
                    pred_ans = 'noanswer'
            else:
                start_token_ = finetune_start(start_token, context_token_ids, tokenizer)
                star_char = context_token_spans[start_token_][0]
                end_char = context_token_spans[end_token][1]
                pred_ans = batch['context'][i][star_char:end_char + 1].strip()
                if start_token_ != start_token:
                    logger.debug(f"finetune predicted answer "
                                 f"『{batch['context'][i][context_token_spans[start_token][0]:end_char + 1]}』->"
                                 f"『{pred_ans}』")

            if nn_input['answer_starts'][i][0] in [-1, NA_POS]:
                gold_answers = ['noanswer']
            else:
                gold_answers = qas_example.get('answers', [qas_example['answer']])
            answer_ems['all'].append(
                max(float(exact_match_score(pred_ans, gold_ans)) for gold_ans in gold_answers)
            )
            answer_f1s['all'].append(
                max(float(f1_score(pred_ans, gold_ans)[0]) for gold_ans in gold_answers)
            )

            link_label = nn_input['link_label'][i].item() if nn_input['link_label'][i].item() != -1 else 0
            link_accuracies['all'].append((pred_link[i] == link_label).float().item())

            gold_action = FUNCTIONS[nn_input['action_label'][i].item()]
            if gold_action == 'ANSWER':
                ans_em = answer_ems['all'][-1]
                ans_f1 = answer_f1s['all'][-1]
                answer_ems['should'].append(ans_em)
                answer_f1s['should'].append(ans_f1)
                answer_ems['act'].append(ans_em * (pred_action[i] == nn_input['action_label'][i]).float().item())
                answer_f1s['act'].append(ans_f1 * (pred_action[i] == nn_input['action_label'][i]).float().item())
                cmd_accuracies.append(answer_f1s['act'][-1])
            elif gold_action == 'LINK':
                link_acc = link_accuracies['all'][-1]
                link_accuracies['should'].append(link_acc)
                link_accuracies['act'].append(link_acc * (pred_action[i] == nn_input['action_label'][i]).float().item())
                cmd_accuracies.append(link_accuracies['act'][-1])
            else:
                cmd_accuracies.append((pred_action[i] == nn_input['action_label'][i]).float().item())

            joint_accuracies.append(
                cmd_accuracies[-1] * collection_f1(para_preds[i].nonzero().squeeze_(1).tolist(),
                                                   nn_input['paras_label'][i].nonzero().squeeze_(1).tolist())
            )

            if FUNCTIONS[pred_action[i].item()] == 'ANSWER':
                answer_ems['practical'].append(answer_ems['all'][-1])
                answer_f1s['practical'].append(answer_f1s['all'][-1])
            elif FUNCTIONS[pred_action[i].item()] == 'LINK':
                link_accuracies['practical'].append(link_accuracies['all'][-1])

    model.train()
    assert len(joint_accuracies) == len(cmd_accuracies) == len(action_accuracies)

    metrics = {
        "obs_acc": np.mean(obs_accuracies) * 100.,
        "para_acc": np.mean(para_accuracies) * 100.,
        "para_f1": np.mean(para_f1_scores) * 100.,
        "sent_acc": np.mean(sent_accuracies) * 100.,
        "sent_f1": np.mean(sent_f1_scores) * 100.,
        "action_acc": np.mean(action_accuracies) * 100.,
        "cmd_acc": np.mean(cmd_accuracies) * 100.,
        "joint_acc": np.mean(joint_accuracies) * 100.
    }
    for k, v in link_accuracies.items():
        metrics[f"link_acc_{k}"] = np.mean(v) * 100. if len(v) > 0 else 0.
    metrics["link_acc"] = harmonic_mean(metrics["link_acc_practical"], metrics[f"link_acc_act"])
    for k, v in answer_ems.items():
        metrics[f"answer_em_{k}"] = np.mean(v) * 100. if len(v) > 0 else 0.
    metrics["answer_em"] = harmonic_mean(metrics["answer_em_practical"], metrics[f"answer_em_act"])
    for k, v in answer_f1s.items():
        metrics[f"answer_f1_{k}"] = np.mean(v) * 100. if len(v) > 0 else 0.
    metrics["answer_f1"] = harmonic_mean(metrics["answer_f1_practical"], metrics[f"answer_f1_act"])

    return metrics


if __name__ == "__main__":
    main()
