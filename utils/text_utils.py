# -*- coding: UTF-8 -*-

# from html import unescape
import logging
# import re
import unicodedata
# from urllib.parse import unquote

from fuzzysearch import find_near_matches
from Levenshtein import distance, ratio

from transformers import ElectraTokenizerFast

from basic_tokenizer import SimpleTokenizer
from config import NA_POS

logger = logging.getLogger(__name__)


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char in [" ", "\t", "\n", "\r"]:
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace characters.
    if char in ["\t", "\n", "\r"]:
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    if char in ["～", "￥", "×"]:
        return True
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if (0x4E00 <= cp <= 0x9FFF or
            0x3400 <= cp <= 0x4DBF or
            0x20000 <= cp <= 0x2A6DF or
            0x2A700 <= cp <= 0x2B73F or
            0x2B740 <= cp <= 0x2B81F or
            0x2B820 <= cp <= 0x2CEAF or
            0xF900 <= cp <= 0xFAFF or
            0x2F800 <= cp <= 0x2FA1F):
        return True

    return False


def is_word_boundary(char):
    return is_whitespace(char) or is_punctuation(char) or is_chinese_char(char)


def clean_text(text):
    # unescaped_text = unescape(text)
    # unquoted_text = unquote(unescaped_text, 'utf-8')
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        if is_whitespace(char):
            output.append(" ")
        # elif char in ["–"]:
        #     output.append("-")
        else:
            output.append(char)
    output_text = ''.join(output)
    # output_text = re.sub(r' {2,}', ' ', output_text).strip()
    return output_text


def norm_text(s):
    return ' '.join(clean_text(s).strip().split())


def normalize_unicode(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def rm_whitespaces(string: str) -> str:
    return ''.join(string.split())


def find_closest_substr(src, tgt, step=1):
    s = src.find(tgt)
    if s >= 0:
        return s, s + len(tgt), 0

    max_l_dist = step
    while True:
        matches = find_near_matches(tgt, src, max_l_dist=max_l_dist)
        if len(matches) > 0:
            min_l_dist = max_l_dist + 1
            best_match = None
            for m in matches:
                if m.dist < min_l_dist:
                    min_l_dist = m.dist
                    best_match = m
            return best_match.start, best_match.end, best_match.dist
        else:
            max_l_dist += step
            if max_l_dist >= len(tgt):
                break

    return -1, -1, len(tgt)


def find_str(string, sub_str, word_boundary=True, ignore_case=False):
    """

    Args:
        string (str):
        sub_str (str):
        word_boundary (bool):
        ignore_case (bool):

    Returns:

    """
    if ignore_case:
        string = string.lower()
        sub_str = sub_str.lower()
    start = string.find(sub_str)
    while start != -1:
        if word_boundary:
            pre_char = string[start - 1] if start - 1 >= 0 else ' '
            next_char = string[start + len(sub_str)] if start + len(sub_str) < len(string) else ' '
            if is_word_boundary(pre_char) and is_word_boundary(next_char):
                return start
        else:
            return start
        start = string.find(sub_str, start + 1)
    return -1


def find_all_str(string, sub_str, word_boundary=True, ignore_case=False):
    starts = []
    offset = 0
    start = find_str(string, sub_str, word_boundary, ignore_case)
    while start >= 0:
        starts.append(offset + start)
        offset += start + len(sub_str)
        if len(string[offset:]) < len(sub_str):
            break
        start = find_str(string[offset:], sub_str, word_boundary, ignore_case)
    return starts


def fuzzy_find_all(source, patterns, tokenizer=None, ignore_case=False, max_l_dist=0, min_ratio=1.0, level='word'):
    """

    Args:
        source (str):
        patterns (list[str]):
        tokenizer:
        ignore_case (bool):
        max_l_dist (int):
        min_ratio (float)
        level (str): word | char

    Returns:
        tuple[list[tuple], list[str]]:
    """
    if tokenizer is None:
        tokenizer = SimpleTokenizer()
    sep = '' if level == 'char' else ' '
    src_tokens = tokenizer.tokenize(source)
    src_words = src_tokens.words(uncased=ignore_case)
    src_offsets = src_tokens.offsets()
    _src_words = [normalize_unicode(w) for w in src_words]
    spans, matches = [], []
    for pattern in patterns:
        pat_tokens = tokenizer.tokenize(pattern)
        pat_words = pat_tokens.words(uncased=ignore_case)
        _pat_words = [normalize_unicode(w) for w in pat_words]
        if len(_pat_words) == 0:
            continue
        _pattern = sep.join(_pat_words)
        for s in range(0, len(_src_words) - len(_pat_words) + 1):
            e = min(s + len(_pat_words), len(_src_words))
            while e + 1 <= len(_src_words) and _pattern.startswith(sep.join(_src_words[s:e + 1])):
                e += 1
            _candidate = sep.join(_src_words[s:e])
            if (start_or_end_with(_pattern, _candidate) and
                    distance(_pattern, _candidate) <= min(max_l_dist, len(_pattern), len(_candidate)) and
                    ratio(_pattern, _candidate) >= min_ratio):
                match = src_tokens.slice(s, e).untokenize()
                span = (src_offsets[s][0], src_offsets[e - 1][1])
                assert source[span[0]:span[1]] == match
                spans.append(span)
                matches.append(match)
    return spans, matches


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        locale.atof(s)
        return True
    except ValueError:
        pass

    try:
        for c in s:
            unicodedata.numeric(c)
        return True
    except (TypeError, ValueError):
        pass
    return False


def atof(s):
    try:
        return float(s)
    except ValueError:
        pass

    try:
        import locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        return locale.atof(s)
    except ValueError:
        pass

    return None


def is_startswith(a: str, b: str) -> bool:
    short, long = sorted([a, b], key=lambda x: len(x))
    return long.startswith(short)


def is_endswith(a: str, b: str) -> bool:
    short, long = sorted([a, b], key=lambda x: len(x))
    return long.endswith(short)


def start_or_end_with(a: str, b: str) -> bool:
    return is_startswith(a, b) or is_endswith(a, b)


def finetune_start(start_idx, token_ids, tokenizer):
    """

    Args:
        start_idx (int):
        token_ids (list[int]):
        tokenizer (ElectraTokenizerFast):

    Returns:

    """
    start_idx_ = start_idx
    while start_idx_ >= 0 and tokenizer.convert_ids_to_tokens(token_ids[start_idx_]).startswith('##'):
        start_idx_ -= 1
    return start_idx_


def predict_answer(pred_start, pred_end, token_ids, context, context_token_spans, context_token_offset, tokenizer):
    start_token = pred_start - context_token_offset
    end_token = pred_end - context_token_offset
    context_token_ids = token_ids[context_token_offset:]
    if start_token < 0:
        if pred_start != pred_end or pred_start not in [1, 2, NA_POS]:
            logger.warning(f"predicted unexpected ans_span [{pred_start}, {pred_end}]")
        if pred_start == 1:
            pred_ans = 'yes'
        elif pred_start == 2:
            pred_ans = 'no'
        else:
            pred_ans = 'noanswer'
    else:
        start_token_ = finetune_start(start_token, context_token_ids, tokenizer)
        star_char = context_token_spans[start_token_][0]
        end_char = context_token_spans[end_token][1]
        pred_ans = context[star_char:end_char + 1].strip()
        if start_token_ != start_token:
            logger.debug(f"finetune predicted answer 『{context[context_token_spans[start_token][0]:end_char + 1]}』"
                         f"->『{pred_ans}』")

    return pred_ans
