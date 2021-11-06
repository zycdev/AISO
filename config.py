import argparse

ADDITIONAL_SPECIAL_TOKENS = {
    "YES": "[unused0]",
    "NO": "[unused1]",
    "SOP": "[unused2]",
    "NONE": "[unused3]"
}
FUNCTIONS = ("ANSWER", "BM25", "MDR", "LINK")
FUNC2ID = {func: idx for idx, func in enumerate(FUNCTIONS)}
NA_POS = 3


def common_args():
    parser = argparse.ArgumentParser()

    # input and output
    parser.add_argument("--corpus_path", type=str, default="data/corpus/hotpot-paragraph-5.tsv")
    parser.add_argument("--train_file", type=str, default="data/hotpot-step-train.jsonl")
    parser.add_argument("--predict_file", type=str, default="data/hotpot-step-dev.jsonl")
    parser.add_argument("--output_dir", default="ckpts", type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")

    parser.add_argument("--do_train", default=False, action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=False, action="store_true",
                        help="for final test submission")

    # model
    parser.add_argument("--encoder_name", default="google/electra-base-discriminator", type=str)
    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")

    # data
    parser.add_argument("--max_seq_len", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_q_len", default=96, type=int)
    parser.add_argument("--max_obs_len", default=256, type=int)
    parser.add_argument("--max_ans_len", default=64, type=int)
    parser.add_argument("--hard_negs_per_state", type=int, default=2,
                        help="how many hard negative observations per state")
    parser.add_argument("--memory_size", type=int, default=3,
                        help="max num of passages stored in memory")
    parser.add_argument("--max_distractors", type=int, default=2,
                        help="max num of distractor passages in context")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--strict", action="store_true", help="whether to strictly use original data of dataset")

    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument("--per_gpu_infer_batch_size", default=16, type=int,
                        help="Batch size per GPU for inference.")
    parser.add_argument("--save-prediction", default="", type=str)

    parser.add_argument("--sp_pred", action="store_true", help="whether to predict sentence sp")

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    return parser


def train_args():
    parser = common_args()

    parser.add_argument('--tag', default=None, type=str,
                        help='The comment to the experiment')
    parser.add_argument('--comment', default=None, type=str,
                        help='The comment to the experiment')

    # model
    parser.add_argument('--cmd_dropout_prob', type=float, default=0.1)

    # optimization
    parser.add_argument("--sp_weight", default=0.0, type=float, help="weight of the sp loss")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU for training.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_ratio", default=0.0, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--use_adam", action="store_true",
                        help="use adam or adamW")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=2.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--log_period', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--eval_period', type=int, default=1000,
                        help="Evaluate every X updates steps.")
    parser.add_argument("--criterion_metric", default="joint_f1")
    parser.add_argument('--save_period', type=int, default=-1,
                        help="Save checkpoint every X updates steps.")

    return parser.parse_args()
