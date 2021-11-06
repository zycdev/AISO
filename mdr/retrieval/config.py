import argparse
from ast import parse
from typing import NamedTuple


class ClusterConfig(NamedTuple):
    dist_backend: str
    dist_url: str


def common_args():
    parser = argparse.ArgumentParser()

    # task
    parser.add_argument("--train_file", type=str, default="../data/nq-with-neg-train.txt")
    parser.add_argument("--predict_file", type=str, default="../data/nq-with-neg-dev.txt")
    parser.add_argument("--num_workers", default=20, type=int)
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set")

    # model
    parser.add_argument("--model_name", default="bert-base-uncased", type=str)
    parser.add_argument("--init_checkpoint", default="", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--max_c_len", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_q_len", default=50, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--max_q_sp_len", default=50, type=int)
    parser.add_argument("--sent_level", action="store_true")
    parser.add_argument("--rnn_retriever", action="store_true")
    parser.add_argument("--predict_batch_size", default=512, type=int, help="Batch size for prediction")
    parser.add_argument("--shared_encoder", action="store_true")

    # multi vector scheme
    parser.add_argument("--multi_vector", type=int, default=1)
    parser.add_argument("--scheme", type=str, default="none", help="how to get the multivector, layerwise or tokenwise")

    # momentum
    parser.add_argument("--momentum", action="store_true")
    parser.add_argument("--init_retriever", type=str, default="")
    parser.add_argument("--k", type=int, default=38400, help="memory bank size")
    parser.add_argument("--m", type=float, default=0.999, help="momentum")

    # NQ multi-hop trial
    parser.add_argument("--nq-multi", action="store_true",
                        help="train the NQ retrieval model to recover from error cases")

    return parser


def train_args():
    parser = common_args()
    # optimization
    parser.add_argument('--prefix', type=str, default="eval")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--output_dir", default="./logs", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=128,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=50, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--save_checkpoints_steps", default=20000, type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--iterations_per_loop", default=1000, type=int,
                        help="How many steps to make in each estimator call.")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument('--seed', type=int, default=3,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--eval_period', type=int, default=2500)
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument("--stop_drop", default=0, type=float)
    parser.add_argument("--use_adam", action="store_true")
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_steps.")

    return parser.parse_args()


def encode_args():
    parser = common_args()
    parser.add_argument('--corpus_file', required=True, type=str, default=None, help='Path to passages .tsv file')
    parser.add_argument("--strict", action="store_true", help="whether to strictly use original data of dataset")
    parser.add_argument('--embedding_prefix', required=True, type=str, default=None,
                        help='Output path(prefix) to write embeddings to')
    parser.add_argument('--num_shards', type=int, default=1, help="Total amount of data shards")
    parser.add_argument('--shard_id', type=int, default=0, help="Number(0-based) of data shard to process")
    args = parser.parse_args()
    assert args.init_checkpoint, 'Please specify --init_checkpoint checkpoint to init model weights'

    return args
