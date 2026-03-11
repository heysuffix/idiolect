"""
Language-agnostic feature extraction.

These features are designed to capture writing style patterns
that persist regardless of which language the author is writing in.
"""

import re
import math
from dataclasses import dataclass
from typing import List


@dataclass
class StyleFeatures:
    # Sentence-level
    sent_length_mean: float
    sent_length_std: float
    sent_length_min: float
    sent_length_max: float

    # Punctuation
    comma_per_sent: float
    exclamation_ratio: float
    question_ratio: float
    ellipsis_density: float
    dash_density: float
    semicolon_density: float
    colon_density: float
    parenthesis_density: float
    quote_density: float

    # Word-level
    word_length_mean: float
    word_length_std: float
    lexical_richness: float       # type-token ratio (normalized)
    avg_word_length_dist: List[float]  # histogram [1-3, 4-6, 7-9, 10+] chars

    # Paragraph
    para_sent_mean: float         # average sentences per paragraph
    para_length_mean: float       # average paragraph word count

    # Misc behavioral
    capitalized_mid_sent: float   # mid-sentence capitalized words ratio
    number_density: float         # digits / total words
    short_sent_ratio: float       # sentences <= 5 words
    long_sent_ratio: float        # sentences >= 25 words

    def to_vector(self) -> List[float]:
        return [
            self.sent_length_mean,
            self.sent_length_std,
            self.sent_length_min,
            self.sent_length_max,
            self.comma_per_sent,
            self.exclamation_ratio,
            self.question_ratio,
            self.ellipsis_density,
            self.dash_density,
            self.semicolon_density,
            self.colon_density,
            self.parenthesis_density,
            self.quote_density,
            self.word_length_mean,
            self.word_length_std,
            self.lexical_richness,
            *self.avg_word_length_dist,
            self.para_sent_mean,
            self.para_length_mean,
            self.capitalized_mid_sent,
            self.number_density,
            self.short_sent_ratio,
            self.long_sent_ratio,
        ]


def _split_sentences(text: str) -> List[str]:
    # Basic multilingual sentence splitter — works on punctuation patterns
    # Handles ., !, ?, and also ... as single unit
    text = re.sub(r'\.{3,}', '…', text)
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _split_paragraphs(text: str) -> List[str]:
    paras = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in paras if p.strip()]


def _tokenize_words(text: str) -> List[str]:
    # Extract alphabetic sequences — language agnostic
    return re.findall(r'[^\s\d\W]+', text, re.UNICODE)


def _count_char(text: str, char: str) -> int:
    return text.count(char)


def extract_features(text: str) -> StyleFeatures:
    if not text or len(text.strip()) < 50:
        raise ValueError("Text too short for reliable feature extraction (min 50 chars)")

    sentences = _split_sentences(text)
    paragraphs = _split_paragraphs(text)
    words = _tokenize_words(text)
    total_words = len(words) or 1
    total_sents = len(sentences) or 1
    total_chars = len(text)

    # Sentence lengths (in words)
    sent_lengths = []
    for s in sentences:
        ww = _tokenize_words(s)
        sent_lengths.append(len(ww))

    sent_arr = sent_lengths or [0]
    sent_mean = sum(sent_arr) / len(sent_arr)
    sent_variance = sum((x - sent_mean) ** 2 for x in sent_arr) / len(sent_arr)
    sent_std = math.sqrt(sent_variance)

    # Punctuation features (per sentence or per 100 words)
    commas = _count_char(text, ',')
    exclamations = sum(1 for s in sentences if s.rstrip().endswith('!'))
    questions = sum(1 for s in sentences if s.rstrip().endswith('?'))
    ellipses = text.count('…') + text.count('...')
    dashes = _count_char(text, '—') + _count_char(text, '–') + _count_char(text, '-')
    semicolons = _count_char(text, ';')
    colons = _count_char(text, ':')
    parens = (text.count('(') + text.count('[')) / 2
    quotes = (text.count('"') + text.count('"') + text.count('"') + text.count("'")) / 2

    scale = 100 / total_words  # per 100 words

    # Word length features
    word_lengths = [len(w) for w in words]
    wl_mean = sum(word_lengths) / total_words
    wl_variance = sum((x - wl_mean) ** 2 for x in word_lengths) / total_words
    wl_std = math.sqrt(wl_variance)

    bins = [0, 0, 0, 0]  # [1-3], [4-6], [7-9], [10+]
    for wl in word_lengths:
        if wl <= 3:
            bins[0] += 1
        elif wl <= 6:
            bins[1] += 1
        elif wl <= 9:
            bins[2] += 1
        else:
            bins[3] += 1
    bins_norm = [b / total_words for b in bins]

    # Lexical richness: normalized type-token ratio
    # Use moving window to normalize for text length
    window = min(100, total_words)
    unique_in_window = len(set(w.lower() for w in words[:window]))
    lexical_richness = unique_in_window / window

    # Paragraph features
    para_sent_counts = []
    para_word_counts = []
    for p in paragraphs:
        ps = _split_sentences(p)
        pw = _tokenize_words(p)
        para_sent_counts.append(len(ps))
        para_word_counts.append(len(pw))

    para_sent_mean = sum(para_sent_counts) / len(para_sent_counts) if para_sent_counts else 1
    para_length_mean = sum(para_word_counts) / len(para_word_counts) if para_word_counts else total_words

    # Mid-sentence capitalized words (indicates proper nouns, emphasis, acronyms)
    all_tokens = re.findall(r'\b\S+\b', text)
    mid_caps = 0
    for i, tok in enumerate(all_tokens[1:], 1):  # skip first word of text
        clean = re.sub(r'[^a-zA-ZÀ-ÿ]', '', tok)
        if clean and clean[0].isupper():
            mid_caps += 1
    cap_ratio = mid_caps / (len(all_tokens) - 1) if len(all_tokens) > 1 else 0

    # Number density
    number_tokens = re.findall(r'\b\d+[\d.,]*\b', text)
    number_density = len(number_tokens) * scale

    # Short/long sentence ratios
    short_sents = sum(1 for sl in sent_lengths if sl <= 5)
    long_sents = sum(1 for sl in sent_lengths if sl >= 25)
    short_ratio = short_sents / total_sents
    long_ratio = long_sents / total_sents

    return StyleFeatures(
        sent_length_mean=sent_mean,
        sent_length_std=sent_std,
        sent_length_min=float(min(sent_arr)),
        sent_length_max=float(min(max(sent_arr), 200)),  # cap outliers
        comma_per_sent=commas / total_sents,
        exclamation_ratio=exclamations / total_sents,
        question_ratio=questions / total_sents,
        ellipsis_density=ellipses * scale,
        dash_density=dashes * scale,
        semicolon_density=semicolons * scale,
        colon_density=colons * scale,
        parenthesis_density=parens * scale,
        quote_density=quotes * scale,
        word_length_mean=wl_mean,
        word_length_std=wl_std,
        lexical_richness=lexical_richness,
        avg_word_length_dist=bins_norm,
        para_sent_mean=para_sent_mean,
        para_length_mean=para_length_mean,
        capitalized_mid_sent=cap_ratio,
        number_density=number_density,
        short_sent_ratio=short_ratio,
        long_sent_ratio=long_ratio,
    )
