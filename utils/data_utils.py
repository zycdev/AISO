from html import unescape
import json
import logging
from tqdm import tqdm
from typing import Callable

from elasticsearch import Elasticsearch, helpers

from utils.text_utils import norm_text

logger = logging.getLogger(__name__)


def get_valid_links(passage: dict, strict: bool = False, title2id_func: Callable = None):
    hyperlinks = {}
    if strict:
        content_start, content_end = passage['sentence_spans'][0][0], passage['sentence_spans'][-1][1]
    else:
        content_start, content_end = 0, len(passage['text'])
    for tgt, mention_spans in passage['hyperlinks'].items():
        if (len(mention_spans) > 0 and
                tgt != unescape(passage['title']) and
                (title2id_func is None or title2id_func(tgt) is not None)):
            valid_mention_spans = []
            for anchor_span in mention_spans:
                if content_start <= anchor_span[0] < anchor_span[1] < content_end:
                    valid_mention_spans.append(anchor_span)
            if len(valid_mention_spans) > 0:
                hyperlinks[tgt] = valid_mention_spans

    return hyperlinks


def load_corpus_(corpus_path, for_hotpot=True, require_hyperlinks=False, index_name='enwiki-20171001-paragraph-3'):
    corpus = dict()
    title2id = dict()
    with open(corpus_path) as f:
        for line in f:
            segs = line.strip().split('\t')
            p_id, text, title = segs[:3]
            p_id, text, title = p_id.strip(), text.strip(), title.strip()
            doc_id = '_'.join(p_id.split('_')[:-1])
            if p_id == 'id':
                continue
            if p_id in corpus:
                logger.warning(f"Duplicate passage id: {p_id} ({title})")
            if unescaped_title in title2id and not title2id[unescaped_title].startswith(doc_id):
                logger.warning(f"Duplicate title: {unescaped_title}")
            corpus[p_id] = {
                "title": title,
                "text": text,
                "sentence_spans": []
            }
            if for_hotpot:
                corpus[p_id]['sentence_spans'] = [tuple(span) for span in eval(segs[3])]
            unescaped_title = unescape(title)
            if for_hotpot:
                title2id[unescaped_title] = p_id  # passage id
            else:
                title2id[unescaped_title] = doc_id  # document id

    if require_hyperlinks:
        es = Elasticsearch(['10.60.0.59:9200'], timeout=30)
        if for_hotpot:
            query = {"query": {"term": {"for_hotpot": True}}}
        else:
            query = {"query": {"match_all": {}}}
        para_num = es.count(index=index_name, body=query)['count']
        for hit in tqdm(helpers.scan(es, query=query, index=index_name), total=para_num):
            para = hit['_source']
            if para['para_id'] in corpus:
                corpus[para['para_id']]['hyperlinks'] = para['hyperlinks']
            else:
                assert para['para_id'][-3:] == '_-1'
                corpus[para['para_id']] = {
                    "title": para['title'],
                    "text": para['text'],
                    "sentence_spans": [],
                    "hyperlinks": para['hyperlinks']
                }

    logger.info(f"Loaded {len(corpus):,d} passages from {corpus_path}")

    return corpus, title2id


def load_corpus(corpus_path, for_hotpot=False, require_hyperlinks=False, preprocess=False):
    corpus = dict()
    title2id = dict()
    logger.info(f"Loading corpus from {corpus_path} ...")
    with open(corpus_path) as f:
        num_field = None
        for line in f:
            segs = [field.strip() for field in line.strip().split('\t')]
            if num_field is None:
                num_field = len(segs)
            elif len(segs) != num_field:
                logger.warning(f'Wrong line format: {segs[0]}')
            if segs[0] == 'id':
                continue

            p_id, text = segs[:2]
            title = segs[2] if len(segs) > 2 else ''
            if text == '' and title == '':
                logger.warning(f"empty passage: {p_id}")
                continue
            hyperlinks = json.loads(segs[3]) if len(segs) > 3 else dict()
            sentence_spans = [tuple(span) for span in eval(segs[4])] if len(segs) > 4 else [(0, len(text))]
            unescaped_title = unescape(title)
            if preprocess:
                text = norm_text(text)

            if p_id in corpus:
                logger.warning(f"Duplicate passage id: {p_id} ({title})")
            corpus[p_id] = {
                "title": title,
                "text": text,
                "sentence_spans": sentence_spans
            }
            if require_hyperlinks:
                corpus[p_id]['hyperlinks'] = {
                    unescape(t): [tuple(a['span']) for a in anchors if a['span'][0] != a['span'][1]]
                    for t, anchors in hyperlinks.items()
                }

            if unescaped_title:
                if for_hotpot:
                    if unescaped_title in title2id:
                        logger.warning(f"Duplicate title: {unescaped_title}")
                    title2id[unescaped_title] = p_id
                elif unescaped_title not in title2id:
                    title2id[unescaped_title] = [p_id]
                else:
                    for other_p_id in title2id[unescaped_title]:
                        if not other_p_id.startswith('_'.join(p_id.split('_')[:-1])):
                            logger.warning(f"Duplicate title: {unescaped_title}")
                    title2id[unescaped_title].append(p_id)
    logger.info(f"Loaded {len(corpus):,d} passages")

    return corpus, title2id


def load_collection(corpus_path, preprocess=False):
    corpus = dict()
    with open(corpus_path) as f:
        for line in f:
            p_id, text = [field.strip() for field in line.strip().split('\t')]
            if p_id == 'id':
                continue
            if p_id in corpus:
                logger.warning(f"Duplicate passage id: {p_id} ({text})")
            if preprocess:
                text = norm_text(text)
            corpus[p_id] = text
    return corpus


def load_qas(file_path):
    qas_samples = []
    with open(file_path) as f:
        for line in f:
            q_id, question, answer, sp_facts = line.strip().split('\t')
            sp_facts = json.loads(sp_facts)
            qas_samples.append((q_id, (question, answer, sp_facts)))
    logger.info(f"Loaded {len(qas_samples):,d} samples from the {file_path}")
    return qas_samples


def load_qrels(file_path):
    qrels = dict()
    with open(file_path) as f:
        for line in f:
            q_id, rels = [field.strip() for field in line.strip().split('\t')]
            qrels[q_id] = json.loads(rels)
    logger.info(f"Loaded {len(qrels):,d} (q_id, rel) pairs from the {file_path}")
    return qrels


def load_queries(file_path):
    queries = dict()
    with open(file_path) as f:
        for line in f:
            q_id, query = [field.strip() for field in line.strip().split('\t')]
            queries[q_id] = query
    logger.info(f"Loaded {len(queries):,d} queries from the {file_path}")
    return queries


def load_samples(file_path, test=True):
    samples = []
    with open(file_path) as f:
        for line in f:
            segs = line.strip().split('\t')
            q_id, question = segs[:2]
            if test:
                samples.append((q_id, (question,)))
            else:
                samples.append((q_id, (question, segs[2], json.loads(segs[3]))))
    logger.info(f"Loaded {len(samples):,d} samples from the {file_path}")
    return samples
