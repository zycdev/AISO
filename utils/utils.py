from collections import MutableMapping
import logging
import os
import re
import random

import numpy as np
import torch

from fuzzysearch import find_near_matches

logger = logging.getLogger(__name__)


def chunk(items, size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def flatten_dict(d, key_prefix='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = key_prefix + sep + k if key_prefix else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def harmonic_mean(a, b):
    if 0 in (a, b):
        return 0.

    return 2 * a * b / (a + b)


def map_span(index_map, span):
    s, e = span  # include end
    assert s <= e, f"[{s}, {e}]"
    if not 0 <= s <= e < len(index_map):
        return -1, -1

    while index_map[s] < 0 and s + 1 <= e:
        s += 1
    ns = index_map[s]
    if ns < 0:
        return -1, -1

    while index_map[e] < 0 and e - 1 >= s:
        e -= 1
    ne = index_map[e]  # include
    assert ns <= ne
    return ns, ne


def find_closest_subseq(sequence, sub_seq, dist_step=1, max_dist=10, min_ratio=0.8):
    if len(sub_seq) == 0 or len(sequence) == 0:
        return -1, -1, 0

    try:
        starts = kmp_find_all(sequence, sub_seq)
    except:
        import pdb
        pdb.set_trace()
        starts = -1
    if len(starts) > 0:
        start = min(starts)
        return start, start + len(sub_seq), 0

    dist_step = max(1, int(dist_step))
    max_dist = min(max_dist, len(sequence), len(sub_seq))
    l_dist = dist_step
    while l_dist <= max_dist:
        matches = find_near_matches(sub_seq, sequence, max_l_dist=l_dist, max_substitutions=0, max_insertions=0)
        if len(matches) > 0:
            min_l_dist = l_dist + 1
            best_match = None
            for m in matches:
                if m.dist < min_l_dist:
                    min_l_dist = m.dist
                    best_match = m
            match_len = best_match.end - best_match.start
            if 1 - best_match.dist / (len(sub_seq) + match_len) < min_ratio:
                return -1, -1, len(sub_seq)
            return best_match.start, best_match.end, best_match.dist
        l_dist += dist_step

    return -1, -1, len(sub_seq)


def has_overlap(src_span, tgt_spans):
    for tgt_span in tgt_spans:
        if not (src_span[0] > tgt_span[1] or src_span[1] < tgt_span[0]):
            return True
    return False


def collection_f1(pred_set, tgt_set):
    if not isinstance(pred_set, set):
        pred_set = set(pred_set)
    if not isinstance(tgt_set, set):
        tgt_set = set(tgt_set)

    if len(pred_set) == 0 and len(tgt_set) == 0:
        return 1.
    elif len(pred_set) == 0 or len(tgt_set) == 0:
        return 0.
    else:
        intersection = pred_set & tgt_set
        # f1 = (2. * len(intersection)) / (len(pred_set) + len(tgt_set))
        if len(intersection) == 0:
            return 0.
        precision = len(intersection) / len(pred_set)
        recall = len(intersection) / len(tgt_set)
        f1 = (2. * precision * recall) / (precision + recall)
        return f1


def span_f1(pred_span, tgt_span):
    assert pred_span[0] <= pred_span[1]
    assert tgt_span[0] <= tgt_span[1]

    if pred_span[0] > tgt_span[1] or pred_span[1] < tgt_span[0]:
        return 0.

    overlap = min(pred_span[1], tgt_span[1]) - max(pred_span[0], tgt_span[0]) + 1.
    assert overlap >= 1.
    precision = overlap / (pred_span[1] - pred_span[0] + 1.)
    recall = overlap / (tgt_span[1] - tgt_span[0] + 1.)
    f1 = (2. * precision * recall) / (precision + recall)

    return f1


def kmp_find_all(source, pattern):
    """KMP search
    Return all the matching position of pattern P in S
    """
    assert len(source) > 0 and len(pattern) > 0

    def partial(p):
        next_table = [0]

        for idx in range(1, len(p)):
            next_idx = next_table[idx - 1]
            while next_idx > 0 and p[next_idx] != p[idx]:
                next_idx = next_table[next_idx - 1]
            next_table.append(next_idx + 1 if p[next_idx] == p[idx] else next_idx)
        return next_table

    partial_table = partial(pattern)
    starts = []

    j = 0
    for i in range(len(source)):
        while j > 0 and source[i] != pattern[j]:
            j = partial_table[j - 1]
        if source[i] == pattern[j]:
            j += 1
        if j == len(pattern):
            starts.append(i - (j - 1))
            j = 0

    return starts


def find_nth(src, tgt, index):
    count = 0
    for idx, x in enumerate(src):
        if x == tgt:
            if count == index:
                return idx
            count += 1
    return -1


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def set_global_logging_level(level=logging.ERROR, prefixes=("",)):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        level: desired level. Optional. Default is logging.ERROR
        prefixes: list of one or more str prefixes to match (e.g. ["transformers", "torch"]). Optional.
            Default is `[""]` to match all active loggers.
            The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^({"|".join(prefixes)})')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
