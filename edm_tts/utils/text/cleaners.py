"""Set of default text cleaners"""
# TODO: pick the cleaner for languages dynamically

import re

from anyascii import anyascii

from .abbreviations import abbreviations_en
from .number_norm import normalize_numbers as en_normalize_numbers
from .time_norm import expand_time_english
from .symbols import SYMBOLS

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")


def expand_abbreviations(text, lang="en"):
    if lang == "en":
        _abbreviations = abbreviations_en

    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text).strip()


def convert_to_ascii(text):
    return anyascii(text)


def remove_aux_symbols(text):
    text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
    return text


def replace_symbols(text, lang="en"):
    """Replace symbols based on the lenguage tag.

    Args:
      text:
       Input text.
      lang:
        Lenguage identifier. ex: "en", "fr", "pt", "ca".

    Returns:
      The modified text
      example:
        input args:
            text: "si l'avi cau, diguem-ho"
            lang: "ca"
        Output:
            text: "si lavi cau, diguemho"
    """
    text = text.replace(";", ",")
    text = text.replace("-", " ") if lang != "ca" else text.replace("-", "")
    text = text.replace(":", ",")
    if lang == "en":
        text = text.replace("&", " and ")
    return text


def basic_lowercase_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def remove_other_symbols(text):
    """Remove symbols that are not in the SYMBOLS list."""
    return "".join([char for char in text if char in SYMBOLS])


def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = convert_to_ascii(text)
    # text = lowercase(text)
    text = expand_time_english(text)
    text = en_normalize_numbers(text)
    text = expand_abbreviations(text)
    text = replace_symbols(text)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    text = remove_other_symbols(text)
    return text
