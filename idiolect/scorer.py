"""
Core scoring engine.

Combines:
1. Feature vector similarity  — structural/syntactic writing habits
2. Semantic embedding similarity — topic-agnostic style patterns

The combination is designed to be robust: a spammer can change their
topic, but they can't easily change their comma rhythm or sentence
length distribution.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np

from .features import StyleFeatures, extract_features
from .embeddings import get_style_embedding, embedding_similarity


# Feature weights — tuned to reduce topic-bleed and amplify structural signals
_FEATURE_WEIGHTS = [
    2.0,  # sent_length_mean      — strong authorship signal
    1.5,  # sent_length_std       — consistency of sentence rhythm
    0.5,  # sent_length_min
    0.5,  # sent_length_max
    1.8,  # comma_per_sent        — punctuation rhythm
    1.2,  # exclamation_ratio
    1.2,  # question_ratio
    1.0,  # ellipsis_density
    1.0,  # dash_density
    1.0,  # semicolon_density
    1.0,  # colon_density
    0.8,  # parenthesis_density
    0.8,  # quote_density
    1.5,  # word_length_mean      — vocabulary complexity
    1.0,  # word_length_std
    1.8,  # lexical_richness      — strong signal
    0.7,  # word_len_dist[0] (short words)
    0.7,  # word_len_dist[1]
    0.7,  # word_len_dist[2]
    0.7,  # word_len_dist[3] (long words)
    1.0,  # para_sent_mean        — structural habit
    0.8,  # para_length_mean
    1.5,  # capitalized_mid_sent  — emphasis style
    0.6,  # number_density
    1.3,  # short_sent_ratio      — tendency toward brevity
    1.3,  # long_sent_ratio       — tendency toward elaboration
]


def _normalize_vector(v: List[float], reference: List[float]) -> List[float]:
    """
    Normalize feature vector relative to a reference.
    Uses tanh normalization to bring all features to [-1, 1].
    """
    result = []
    for a, b in zip(v, reference):
        mid = (a + b) / 2 if (a + b) != 0 else 1
        scale = abs(mid) if mid != 0 else 1
        diff = (a - b) / scale
        result.append(diff)
    return result


def _weighted_feature_similarity(feat_a: StyleFeatures, feat_b: StyleFeatures) -> Tuple[float, List[Tuple[str, float, float, float]]]:
    """
    Returns (similarity_score, detailed_breakdown).
    Score is 0-1.
    """
    vec_a = feat_a.to_vector()
    vec_b = feat_b.to_vector()

    feature_names = [
        "sentence length (mean)", "sentence length (variation)",
        "sentence length (min)", "sentence length (max)",
        "comma rhythm", "exclamation use", "question use",
        "ellipsis use", "dash use", "semicolon use", "colon use",
        "parenthesis use", "quote use",
        "word length (mean)", "word length (variation)",
        "lexical richness",
        "short words ratio", "medium words ratio", "long words ratio", "very long words ratio",
        "paragraph density", "paragraph length",
        "mid-sentence capitalization", "number usage",
        "short sentence ratio", "long sentence ratio",
    ]

    weights = _FEATURE_WEIGHTS
    total_weight = 0.0
    weighted_sim = 0.0
    details = []

    for i, (a, b, w, name) in enumerate(zip(vec_a, vec_b, weights, feature_names)):
        magnitude = max(abs(a), abs(b), 0.0001)
        diff = abs(a - b) / magnitude
        sim = math.exp(-2.0 * diff)

        # Zero-zero matches are uninformative — both authors not using a feature
        # tells us little about shared authorship. Penalize these.
        if abs(a) < 0.001 and abs(b) < 0.001:
            sim = 0.5  # neutral, not a positive signal
            w = w * 0.3  # reduce weight of uninformative features

        weighted_sim += sim * w
        total_weight += w
        details.append((name, a, b, sim))

    score = weighted_sim / total_weight
    return score, details


@dataclass
class AttributionResult:
    score: float                    # 0-100 final attribution score
    confidence: str                 # "low" | "medium" | "high" | "very high"
    verdict: str                    # Human-readable conclusion
    feature_score: float            # 0-1 structural similarity
    embedding_score: float          # 0-1 semantic style similarity
    top_matches: List[Tuple[str, float, float, float]]    # (feature, val_a, val_b, similarity)
    top_mismatches: List[Tuple[str, float, float, float]]
    text_a_lang: str
    text_b_lang: str
    warnings: List[str] = None
    ci_low: Optional[float] = None   # 95% confidence interval lower bound
    ci_high: Optional[float] = None  # 95% confidence interval upper bound

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def __str__(self) -> str:
        bar = "█" * int(self.score / 5) + "░" * (20 - int(self.score / 5))
        return (
            f"\n  Score:      {self.score:.1f}/100  [{bar}]\n"
            f"  Confidence: {self.confidence}\n"
            f"  Verdict:    {self.verdict}\n"
            f"\n  Structural similarity:  {self.feature_score * 100:.1f}%\n"
            f"  Semantic style match:   {self.embedding_score * 100:.1f}%\n"
        )


def _detect_language(text: str) -> str:
    """Detect language using langdetect with fallback to heuristic."""
    try:
        from langdetect import detect
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0  # deterministic results
        return detect(text[:2000])  # first 2000 chars is enough
    except Exception:
        pass

    # Fallback: simple function word heuristic
    text_lower = text.lower()
    hints = {
        "tr": ["ve ", "bir ", "bu ", "için ", "ile ", "da ", "de ", "değil"],
        "de": [" die ", " der ", " das ", " und ", " ist ", " nicht "],
        "fr": [" le ", " la ", " les ", " est ", " pas ", " une "],
        "es": [" el ", " la ", " los ", " que ", " con ", " una "],
        "ru": ["что ", "это ", "как ", "для ", "при ", "его "],
        "ar": ["في ", "من ", "على ", "أن ", "التي ", "هذا "],
        "zh": ["的", "是", "了", "在", "有", "不"],
        "en": [" the ", " and ", " that ", " this ", " with ", " not "],
    }
    scores = {lang: sum(1 for w in words if w in text_lower) for lang, words in hints.items()}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "unknown"


# Languages with naturally longer words due to morphology (agglutinative/synthetic)
# Word length features are less reliable as cross-lingual signals for these pairs
_AGGLUTINATIVE = {"tr", "fi", "hu", "et", "az", "kz", "uz", "tk"}


def compare(
    text_a: str,
    text_b: str,
    use_embeddings: bool = True,
    verbose: bool = False,
    quiet: bool = False,
) -> AttributionResult:
    """
    Compare two texts for common authorship.

    Args:
        text_a: First text sample (any language)
        text_b: Second text sample (any language)
        use_embeddings: If False, use only structural features (faster, no model needed)
        verbose: Print progress to stderr
        quiet: Suppress all stderr output (useful for JSON/programmatic use)

    Returns:
        AttributionResult with score and detailed breakdown
    """

    if verbose and not quiet:
        print("[idiolect] Extracting structural features...")

    feat_a = extract_features(text_a)
    feat_b = extract_features(text_b)
    feature_score, details = _weighted_feature_similarity(feat_a, feat_b)

    lang_a = _detect_language(text_a)
    lang_b = _detect_language(text_b)

    # Warn when comparing agglutinative vs analytic language: word length features are biased
    agglutinative_pair = (lang_a in _AGGLUTINATIVE) != (lang_b in _AGGLUTINATIVE)

    if use_embeddings:
        if verbose and not quiet:
            print("[idiolect] Computing multilingual style embeddings...")
        emb_a = get_style_embedding(text_a, verbose=verbose, quiet=quiet)
        emb_b = get_style_embedding(text_b, verbose=verbose, quiet=quiet)
        raw_emb_sim = embedding_similarity(emb_a, emb_b)
        # Normalize from [-1,1] to [0,1]
        emb_score = (raw_emb_sim + 1) / 2
    else:
        emb_score = feature_score  # fallback

    # Combined score: structural features carry more weight for cross-lingual
    if lang_a != lang_b and lang_a != "unknown" and lang_b != "unknown":
        # Cross-lingual: structural features are more reliable than topic-based embeddings
        combined = (feature_score * 0.65) + (emb_score * 0.35)
    else:
        # Same language: embeddings can capture more
        combined = (feature_score * 0.50) + (emb_score * 0.50)

    score = round(combined * 100, 1)

    # Confidence: based on text lengths
    words_a = len(text_a.split())
    words_b = len(text_b.split())
    min_words = min(words_a, words_b)

    if min_words >= 500:
        confidence = "high"
    elif min_words >= 200:
        confidence = "medium"
    elif min_words >= 80:
        confidence = "low"
    else:
        confidence = "very low"

    # Verdict
    if score >= 72:
        verdict = "Strong evidence of common authorship"
    elif score >= 60:
        verdict = "Moderate evidence of common authorship"
    elif score >= 46:
        verdict = "Inconclusive — insufficient stylistic overlap"
    elif score >= 32:
        verdict = "Moderate evidence of different authors"
    else:
        verdict = "Strong evidence of different authors"

    # Top matching and mismatching features
    details_sorted = sorted(details, key=lambda x: x[3], reverse=True)
    top_matches = details_sorted[:5]
    top_mismatches = details_sorted[-5:]

    warnings = []
    if agglutinative_pair:
        warnings.append(
            "One language is agglutinative — word length features may be less reliable. "
            "Use more text samples for higher confidence."
        )

    return AttributionResult(
        score=score,
        confidence=confidence,
        verdict=verdict,
        feature_score=feature_score,
        embedding_score=emb_score,
        top_matches=top_matches,
        top_mismatches=top_mismatches,
        text_a_lang=lang_a,
        text_b_lang=lang_b,
        warnings=warnings,
    )


def _split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.split()) >= 3]


def compare_with_ci(
    text_a: str,
    text_b: str,
    n_bootstrap: int = 100,
    sample_ratio: float = 0.7,
    seed: int = 42,
    verbose: bool = False,
) -> AttributionResult:
    """
    Compare two texts and compute 95% bootstrap confidence intervals.

    Repeatedly samples subsets of sentences from both texts and runs
    structural feature comparison. Returns mean score with CI bounds.

    Args:
        text_a: First text sample
        text_b: Second text sample
        n_bootstrap: Number of bootstrap iterations (default 100)
        sample_ratio: Fraction of sentences to sample each iteration (default 0.7)
        seed: Random seed for reproducibility
        verbose: Show progress

    Returns:
        AttributionResult with ci_low and ci_high populated
    """
    rng = random.Random(seed)

    sents_a = _split_into_sentences(text_a)
    sents_b = _split_into_sentences(text_b)

    if len(sents_a) < 4 or len(sents_b) < 4:
        # Not enough sentences for bootstrap — fall back to regular compare
        result = compare(text_a, text_b, use_embeddings=False)
        result.warnings.append(
            "Too few sentences for bootstrap CI (need 4+). Single-sample score shown."
        )
        return result

    k_a = max(3, int(len(sents_a) * sample_ratio))
    k_b = max(3, int(len(sents_b) * sample_ratio))

    scores = []
    for i in range(n_bootstrap):
        sample_a = " ".join(rng.sample(sents_a, k_a))
        sample_b = " ".join(rng.sample(sents_b, k_b))
        try:
            feat_a = extract_features(sample_a)
            feat_b = extract_features(sample_b)
            fs, _ = _weighted_feature_similarity(feat_a, feat_b)
            scores.append(fs * 100)
        except ValueError:
            continue

        if verbose and (i + 1) % 25 == 0:
            print(f"  Bootstrap: {i + 1}/{n_bootstrap}")

    if not scores:
        result = compare(text_a, text_b, use_embeddings=False)
        result.warnings.append("Bootstrap sampling failed. Single-sample score shown.")
        return result

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))

    # 95% CI using normal approximation
    z = 1.96
    ci_low = max(0.0, round(mean_score - z * std_score, 1))
    ci_high = min(100.0, round(mean_score + z * std_score, 1))

    # Get full result for verdict/confidence/details, use bootstrap mean as score
    result = compare(text_a, text_b, use_embeddings=False)
    result.score = round(mean_score, 1)
    result.ci_low = ci_low
    result.ci_high = ci_high

    # Re-evaluate verdict with bootstrap mean
    if result.score >= 72:
        result.verdict = "Strong evidence of common authorship"
    elif result.score >= 60:
        result.verdict = "Moderate evidence of common authorship"
    elif result.score >= 46:
        result.verdict = "Inconclusive — insufficient stylistic overlap"
    elif result.score >= 32:
        result.verdict = "Moderate evidence of different authors"
    else:
        result.verdict = "Strong evidence of different authors"

    return result
