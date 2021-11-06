# -*- coding: UTF-8 -*-

import re
import unicodedata
from html import unescape
from urllib.parse import unquote


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
