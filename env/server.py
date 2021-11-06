from argparse import Namespace
import logging

from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from redis import Redis

import torch
from transformers import AutoConfig, AutoTokenizer

from drqa.reader import Predictor
import faiss
from mdr.retrieval.models.retriever import RobertaCtxEncoder

from agent.core import Agent
from dense_encoder import MDREncoder
from dense_indexer import DenseHNSWFlatIndexer, DenseFlatIndexer
from config import ADDITIONAL_SPECIAL_TOKENS
from env.core import Environment
from models.union_model import UnionModel
from retriever import SparseRetriever, DenseRetriever
from utils.data_utils import load_corpus
from utils.model_utils import load_state
from utils.utils import set_seed, set_global_logging_level

logger = logging.getLogger()
# if logger.hasHandlers():
#     logger.handlers.clear()
# console = logging.StreamHandler()
# logger.addHandler(console)

faiss.omp_set_num_threads(1)

app = Flask(__name__)
api = Api(app)


@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')


@app.route('/title2id', methods=['POST'])
def title2id():
    return jsonify(env.title2id(**request.get_json(force=True)))


class Passage(Resource):
    @staticmethod
    def get(p_id):
        return jsonify(env.get(p_id))


class Execution(Resource):
    @staticmethod
    def post():
        return jsonify(env.step(**request.get_json(force=True)))


class State(Resource):
    @staticmethod
    def delete():
        env.reset()


class Memory(Resource):
    @staticmethod
    def get(game_id):
        dtype = request.args.get('dtype', default='dict')
        if dtype == 'ordered_passages':
            return jsonify([agent.env.get(p_id) for p_id in
                            sorted(agent.memory[game_id].keys(), key=lambda k: -agent.memory[game_id][k])])
        else:
            return jsonify(agent.memory[game_id])


class Evidence(Resource):
    @staticmethod
    def post(game_id, p_id):
        if p_id not in agent.memory[game_id]:
            agent.memory[game_id][p_id] = 1.0

    @staticmethod
    def delete(game_id, p_id):
        if p_id in agent.memory[game_id]:
            agent.memory[game_id].pop(p_id)


class Action(Resource):
    @staticmethod
    def post():
        return jsonify(agent.act(**request.get_json(force=True), disable_tqdm=True))


class Proposals(Resource):
    @staticmethod
    def get(game_id, step):
        try:
            step = int(step)
        except:
            return jsonify(None)
        if len(agent.proposals[game_id]) == 0 or step >= len(agent.proposals[game_id]):
            return jsonify(None)
        return jsonify(agent.proposals[game_id][step])


class Game(Resource):
    @staticmethod
    def delete():
        agent.reset()


api.add_resource(Passage, '/passages/<string:p_id>')
api.add_resource(Execution, '/executions')
api.add_resource(State, '/states')
api.add_resource(Memory, '/memory/<string:game_id>')
api.add_resource(Evidence, '/memory/<string:game_id>/<string:p_id>')
api.add_resource(Action, '/actions')
api.add_resource(Proposals, '/proposals/<string:game_id>/<string:step>')
api.add_resource(Game, '/games')

if __name__ == '__main__':
    logging.basicConfig(
        format='[%(asctime)s %(levelname)s %(name)s] %(message)s', datefmt='%m/%d %H:%M:%S',
        level=logging.INFO
    )
    logger.setLevel(logging.INFO)
    set_global_logging_level(logging.WARNING, ["elasticsearch"])

    set_seed(0)

    # ########## Load sparse query generator ##########
    sparse_retriever = SparseRetriever('enwiki-20171001-paragraph-5', ['10.208.57.33:9201'], max_retries=4, timeout=15)
    qg1 = Predictor(model='ckpts/golden-retriever/hop1.mdl', embedding_file='data/glove.840B.300d.txt',
                    tokenizer=None, num_workers=-1)
    qg1.cuda()
    qg2 = Predictor(model='ckpts/golden-retriever/hop2.mdl', embedding_file='data/glove.840B.300d.txt',
                    tokenizer=None, num_workers=-1)
    qg2.cuda()

    # ########## Load dense indexer and dense encoder ##########
    dr_config = Namespace(**{
        "model_name": "roberta-base",
        "model_path": "ckpts/mdr/q_encoder.pt",
        "index_prefix_path": "data/index/mdr/hotpot-paragraph-q-strict",  # .hnsw
        "index_buffer_size": 50000,
        "max_q_len": 70,
        "max_q_sp_len": 350
    })
    model_config = AutoConfig.from_pretrained(dr_config.model_name)
    dense_tokenizer = AutoTokenizer.from_pretrained(dr_config.model_name)
    dense_model = RobertaCtxEncoder(model_config, dr_config)
    dense_model = load_state(dense_model, dr_config.model_path, exact=False)
    dense_model.to(torch.device('cuda:1'))
    dense_model.eval()
    dense_encoder = MDREncoder(dense_model, dense_tokenizer, max_p_len=dr_config.max_q_len)
    vector_size = model_config.hidden_size
    if dr_config.index_prefix_path.endswith('hnsw'):
        dense_indexer = DenseHNSWFlatIndexer(vector_size, dr_config.index_buffer_size)
    else:
        dense_indexer = DenseFlatIndexer(vector_size, dr_config.index_buffer_size)
    dense_indexer.deserialize_from(dr_config.index_prefix_path)
    dense_retriever = DenseRetriever(dense_indexer, dense_encoder)

    # ########## Load corpus ##########
    corpus, title2id = load_corpus('data/corpus/hotpot-paragraph-5.tsv', for_hotpot=True, require_hyperlinks=True)

    rds_cfg = {
        "host": "10.60.1.74",  # 10.208.63.53
        "port": 6379,
        "password": "redis4zyc",
        "dbs": {
            "query": 2,
            "bm25": 3,  # 0
            "mdr": 4  # 1
        }
    }

    # ########## Env: WikiWorld ##########
    bm25_redis = Redis(host=rds_cfg['host'], port=rds_cfg['port'], password=rds_cfg['password'],
                       db=rds_cfg['dbs']['bm25'], decode_responses=True)
    mdr_redis = Redis(host=rds_cfg['host'], port=rds_cfg['port'], password=rds_cfg['password'],
                      db=rds_cfg['dbs']['mdr'], decode_responses=True)
    env = Environment(corpus, title2id, sparse_retriever, dense_retriever, bm25_redis, mdr_redis,
                      for_hotpot=True, strict=True, max_ret_size=1000)

    # ########## QA agent ##########
    # backbone = 'google/electra-large-discriminator'
    # init_checkpoint = 'ckpts/td5-exp1-ila.4_electra-large-discriminator_DP0.5_HN2_M2_D2_adamW_SP0.5_B32_LR2.0e-05_' \
    #                   'WU0.1_E30_S42_04270247_pld-sb0-wo*-cmd10/checkpoint_joint.pt'
    backbone = 'google/electra-base-discriminator'
    init_checkpoint = 'ckpts/td5-exp1-ila.4_electra-base-discriminator_DP0.5_HN2_M2_D2_adamW_SP0.5_B32_LR2.0e-05_' \
                      'WU0.1_E30_S42_04202303_pld-sb0-wo*-cmd10/checkpoint_68000.pt'
    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True,
                                              additional_special_tokens=list(ADDITIONAL_SPECIAL_TOKENS.values()))
    union_model = UnionModel(backbone, max_ans_len=64)
    device = torch.device("cuda:1")
    union_model = load_state(union_model, init_checkpoint)
    union_model.to(device)
    union_model.eval()
    query_redis = Redis(host=rds_cfg['host'], port=rds_cfg['port'], password=rds_cfg['password'],
                        db=rds_cfg['dbs']['query'], decode_responses=True)
    agent = Agent(tokenizer, union_model, qg1, qg2, device, env, query_redis,
                  func_mask=(1, 1, 1, 1), memory_size=2, max_seq_len=512, max_q_len=96, max_obs_len=256, strict=True,
                  gold_qas_map=None, oracle_belief=False, oracle_state2action=None)

    app.run(host='0.0.0.0', port=17101, debug=False)
