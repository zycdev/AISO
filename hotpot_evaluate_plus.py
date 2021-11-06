import logging
import sys
import ujson as json
import re
import string
from collections import Counter, defaultdict

from prettytable import PrettyTable, MARKDOWN

logger = logging.getLogger(__name__)


def normalize_answer(s):
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def norm_prediction(prediction):
    for _id, ans in prediction['answer'].items():
        prediction['answer'][_id] = normalize_answer(ans)
    for _id, sp in prediction['sp'].items():
        prediction['sp'][_id] = sorted(sp)
    return prediction


def norm_prediction_file(pred_path, norm_pred_path):
    with open(pred_path, 'r') as f:
        prediction = norm_prediction(json.load(f))
    with open(norm_pred_path, 'w') as f:
        json.dump(prediction, f, sort_keys=True, indent=2)


def predictions2samples(predictions):
    id2as = defaultdict(dict)
    for _id, ans in predictions['answer'].items():
        id2as[_id]['answer'] = ans
    for _id, sp in predictions['sp'].items():
        id2as[_id]['supporting_facts'] = sp
    samples = []
    for _id, ansp in id2as.items():
        samples.append({"_id": _id, "answer": ansp['answer'], "supporting_facts": ansp['supporting_facts']})
    return samples


def f1_score(pred_ans: str, golden_ans: str):
    norm_pred_ans = normalize_answer(pred_ans)
    norm_golden_ans = normalize_answer(golden_ans)

    zero_metric = (0., 0., 0.)

    if norm_pred_ans in ['yes', 'no', 'noanswer'] and norm_pred_ans != norm_golden_ans:
        return zero_metric
    if norm_golden_ans in ['yes', 'no', 'noanswer'] and norm_pred_ans != norm_golden_ans:
        return zero_metric

    pred_tokens = norm_pred_ans.split()
    golden_tokens = norm_golden_ans.split()
    common = Counter(pred_tokens) & Counter(golden_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(golden_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(pred_ans: str, golden_ans: str):
    return normalize_answer(pred_ans) == normalize_answer(golden_ans)


def update_answer(metrics, pred_ans, golden_ans, prefix=''):
    em = exact_match_score(pred_ans, golden_ans)
    f1, prec, recall = f1_score(pred_ans, golden_ans)
    metrics[prefix + 'em'] += float(em)
    metrics[prefix + 'f1'] += f1
    metrics[prefix + 'prec'] += prec
    metrics[prefix + 'recall'] += recall
    return em, f1, prec, recall


def update_sp(metrics, pred_sp_facts, gold_sp_facts, prefix='sp_'):
    pred_sp_sentences = set(map(tuple, pred_sp_facts))
    golden_sp_sentences = set(map(tuple, gold_sp_facts))
    tp, fp, fn = 0, 0, 0
    for e in pred_sp_sentences:
        if e in golden_sp_sentences:
            tp += 1
        else:
            fp += 1
    for e in golden_sp_sentences:
        if e not in pred_sp_sentences:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics[prefix + 'em'] += em
    metrics[prefix + 'f1'] += f1
    metrics[prefix + 'prec'] += prec
    metrics[prefix + 'recall'] += recall
    return em, f1, prec, recall


def update_sp_para(metrics, pred_sp_paras, gold_sp_facts):
    golden_sp_docs = set([sp_fact[0] for sp_fact in gold_sp_facts])
    tp, fp, fn = 0, 0, 0
    for e in pred_sp_paras:
        if e in golden_sp_docs:
            tp += 1
        else:
            fp += 1
    for e in golden_sp_docs:
        if e not in pred_sp_paras:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['spp_em'] += em
    metrics['spp_f1'] += f1
    metrics['spp_prec'] += prec
    metrics['spp_recall'] += recall
    return em, f1, prec, recall


def pretty_metrics(metrics):
    tb = PrettyTable(["", "EM", "F1", "Prec", "Recall"])
    tb.set_style(MARKDOWN)
    tb.align = 'r'
    tb.float_format = ".2"
    tb.add_row(["Answer", metrics["em"], metrics["f1"], metrics["prec"], metrics["recall"]])
    if 'norm_em' in metrics:
        tb.add_row(["Norm answer", metrics["norm_em"], metrics["norm_f1"],
                    metrics["norm_prec"], metrics["norm_recall"]])
    tb.add_row(["Support sentence", metrics["sp_em"], metrics["sp_f1"], metrics["sp_prec"], metrics["sp_recall"]])
    if '_sp_em' in metrics:
        tb.add_row(["Support sentence", metrics["_sp_em"], metrics["_sp_f1"],
                    metrics["_sp_prec"], metrics["_sp_recall"]])
    tb.add_row(["Support passage", metrics["spp_em"], metrics["spp_f1"], metrics["spp_prec"], metrics["spp_recall"]])
    tb.add_row(["Joint", metrics["joint_em"], metrics["joint_f1"], metrics["joint_prec"], metrics["joint_recall"]])
    return tb.get_string()


def show_delta_metrics(new_metrics, base_metrics):
    delta_metrics = {k: new_metrics[k] - base_metrics[k] for k in new_metrics.keys()}
    tb = PrettyTable(["Δ metrics", "EM", "F1", "Prec", "Recall"])
    tb.align = 'r'
    tb.float_format = ".2"
    tb.add_row(["Answer", delta_metrics["em"],
                delta_metrics["f1"], delta_metrics["prec"], delta_metrics["recall"]])
    tb.add_row(["Support sentence", delta_metrics["sp_em"],
                delta_metrics["sp_f1"], delta_metrics["sp_prec"], delta_metrics["sp_recall"]])
    tb.add_row(["Support paragraph", delta_metrics["spp_em"],
                delta_metrics["spp_f1"], delta_metrics["spp_prec"], delta_metrics["spp_recall"]])
    tb.add_row(["Joint", delta_metrics["joint_em"],
                delta_metrics["joint_f1"], delta_metrics["joint_prec"], delta_metrics["joint_recall"]])
    print(tb)


def evaluate(pred_results, gold_samples):
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
               'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
               'spp_em': 0, 'spp_f1': 0, 'spp_prec': 0, 'spp_recall': 0,
               'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    if 'norm_answer' in pred_results:
        metrics.update({'norm_em': 0, 'norm_f1': 0, 'norm_prec': 0, 'norm_recall': 0})
    if '_sp' in pred_results:
        metrics.update({'_sp_em': 0, '_sp_f1': 0, '_sp_prec': 0, '_sp_recall': 0})
    for sample in gold_samples:
        cur_id = sample['_id']
        can_eval_joint = True

        if cur_id not in pred_results['answer']:
            # logger.warning('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, _, prec, recall = update_answer(metrics, pred_results['answer'][cur_id], sample['answer'])

        if 'norm_answer' not in pred_results or cur_id not in pred_results['norm_answer']:
            pass
            # logger.warning('missing norm answer {}'.format(cur_id))
        else:
            update_answer(metrics, pred_results['norm_answer'][cur_id], sample['answer'], 'norm_')

        if cur_id not in pred_results['sp']:
            # logger.warning('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, _, sp_prec, sp_recall = update_sp(metrics, pred_results['sp'][cur_id], sample['supporting_facts'])

        if '_sp' not in pred_results or cur_id not in pred_results['_sp']:
            pass
            # logger.warning('missing sp fact {}'.format(cur_id))
        else:
            update_sp(metrics, pred_results['_sp'][cur_id], sample['supporting_facts'], '_sp_')

        if 'spp' not in pred_results or cur_id not in pred_results['spp']:
            # logger.warning('missing sp paragraph {}'.format(cur_id))
            pred_sp_paras = set([sp_fact[0]
                                 for sp_fact in pred_results['sp'][cur_id]]) if cur_id in pred_results['sp'] else set()
        else:
            pred_sp_paras = set(pred_results['spp'][cur_id])
        update_sp_para(metrics, pred_sp_paras, sample['supporting_facts'])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em
            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    num_sample = len(gold_samples)
    for k in metrics.keys():
        metrics[k] /= num_sample
        metrics[k] *= 100.

    # logger.info('***** Evaluation metrics *****\n' + pretty_metrics(metrics))

    return metrics


def evaluate_subset(pred_results, raw_results, gold_samples):
    sampled_ids = set([raw_result.id for raw_result in raw_results])
    sub_gold_samples = [sample for sample in gold_samples if sample['_id'] in sampled_ids]
    return evaluate(pred_results, sub_gold_samples)


def evaluate_sample(pred_results, sample):
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
               'norm_em': 0, 'norm_f1': 0, 'norm_prec': 0, 'norm_recall': 0,
               'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
               'spp_em': 0, 'spp_f1': 0, 'spp_prec': 0, 'spp_recall': 0,
               'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    cur_id = sample['_id']
    can_eval_joint = True

    if cur_id not in pred_results['answer']:
        logger.warning(f'missing answer {cur_id}')
        can_eval_joint = False
    else:
        em, f1, prec, recall = update_answer(metrics, pred_results['answer'][cur_id], sample['answer'])

    if cur_id not in pred_results['sp']:
        logger.warning(f'missing sp fact {cur_id}')
        can_eval_joint = False
    else:
        sp_em, f1, sp_prec, sp_recall = update_sp(metrics, pred_results['sp'][cur_id], sample['supporting_facts'])

    if cur_id not in pred_results['spp']:
        logger.warning(f'missing sp paragraph {cur_id}')
        pred_sp_paras = set([sp_fact[0] for sp_fact in pred_results['sp'][cur_id]])
    else:
        pred_sp_paras = set(pred_results['spp'][cur_id])
    update_sp_para(metrics, pred_sp_paras, sample['supporting_facts'])

    if can_eval_joint:
        joint_prec = prec * sp_prec
        joint_recall = recall * sp_recall
        if joint_prec + joint_recall > 0:
            joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
        else:
            joint_f1 = 0.
        joint_em = em * sp_em
        metrics['joint_em'] += joint_em
        metrics['joint_f1'] += joint_f1
        metrics['joint_prec'] += joint_prec
        metrics['joint_recall'] += joint_recall

    return metrics


def evaluate_file(pred_file, gold_file):
    with open(pred_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)
    evaluate(prediction, gold)


if __name__ == '__main__':
    evaluate_file(sys.argv[1], sys.argv[2])
