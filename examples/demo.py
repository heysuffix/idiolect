"""
idiolect Python API demo.

Shows how to use idiolect programmatically — useful for integrating
into larger OSINT pipelines or research tools.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from idiolect import compare, extract_features


# ── Sample texts ────────────────────────────────────────────────────────────

AUTHOR_A_ENGLISH = open(os.path.join(os.path.dirname(__file__), "author_a_english.txt")).read()
AUTHOR_A_TURKISH = open(os.path.join(os.path.dirname(__file__), "author_a_turkish.txt")).read()
AUTHOR_B_ENGLISH = open(os.path.join(os.path.dirname(__file__), "author_b_english.txt")).read()
AUTHOR_B_GERMAN  = open(os.path.join(os.path.dirname(__file__), "author_b_german.txt")).read()


# ── 1. Basic comparison ──────────────────────────────────────────────────────

print("=" * 60)
print("Demo 1: Same author — English vs Turkish")
print("=" * 60)

result = compare(AUTHOR_A_ENGLISH, AUTHOR_A_TURKISH)
print(f"Score:       {result.score}/100")
print(f"Verdict:     {result.verdict}")
print(f"Confidence:  {result.confidence}")
print(f"Languages:   {result.text_a_lang} → {result.text_b_lang}")
print()


# ── 2. Different authors ─────────────────────────────────────────────────────

print("=" * 60)
print("Demo 2: Different authors — English vs German")
print("=" * 60)

result2 = compare(AUTHOR_A_ENGLISH, AUTHOR_B_GERMAN)
print(f"Score:       {result2.score}/100")
print(f"Verdict:     {result2.verdict}")
print()


# ── 3. Fast mode (no ML model) ───────────────────────────────────────────────

print("=" * 60)
print("Demo 3: Structural features only (fast, no model)")
print("=" * 60)

result3 = compare(AUTHOR_A_ENGLISH, AUTHOR_A_TURKISH, use_embeddings=False)
print(f"Score:       {result3.score}/100  (structural only)")
print()


# ── 4. Feature inspection ────────────────────────────────────────────────────

print("=" * 60)
print("Demo 4: Style feature extraction")
print("=" * 60)

feat = extract_features(AUTHOR_A_ENGLISH)
print(f"Sentence length:     {feat.sent_length_mean:.1f} words (±{feat.sent_length_std:.1f})")
print(f"Lexical richness:    {feat.lexical_richness:.3f}")
print(f"Comma rhythm:        {feat.comma_per_sent:.3f} per sentence")
print(f"Short sentence ratio:{feat.short_sent_ratio:.3f}")
print(f"Exclamation ratio:   {feat.exclamation_ratio:.3f}")
print()


# ── 5. Bulk comparison ───────────────────────────────────────────────────────

print("=" * 60)
print("Demo 5: Rank suspects against known author")
print("=" * 60)

suspects = {
    "Author A (Turkish)": AUTHOR_A_TURKISH,
    "Author B (English)": AUTHOR_B_ENGLISH,
    "Author B (German)":  AUTHOR_B_GERMAN,
}

results = []
for name, text in suspects.items():
    r = compare(AUTHOR_A_ENGLISH, text)
    results.append((name, r.score, r.verdict))

results.sort(key=lambda x: x[1], reverse=True)

print(f"Known author: Author A (English)\n")
print(f"{'Rank':<5} {'Suspect':<25} {'Score':<8} Verdict")
print("-" * 70)
for i, (name, score, verdict) in enumerate(results, 1):
    print(f"{i:<5} {name:<25} {score:<8.1f} {verdict}")
