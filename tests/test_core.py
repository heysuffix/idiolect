"""
Core tests for idiolect.
Run: python -m pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from idiolect.features import extract_features
from idiolect.scorer import compare


AUTHOR_A_EN = """
The situation in Eastern Europe has, once again, exposed the fundamental weakness
of multilateral institutions. Nobody should be surprised — this has been coming
for years. The patterns were there: the slow erosion of trust, the quiet withdrawal
from commitments, the diplomatic language that masks the absence of any real position.

What's striking is not the crisis itself, but the response. Or rather, the non-response.
Officials speak carefully, choosing words that commit them to nothing. Press releases are
issued. Meetings are scheduled. And meanwhile, on the ground, the reality shifts irreversibly.

I've spoken with people who were there. Not officials — actual people, living through it.
Their accounts don't match the briefings. They never do. There's always this gap between
what institutions say and what individuals experience, and that gap is where the truth lives.
"""

AUTHOR_A_TR = """
Doğu Avrupa'daki durum, bir kez daha, çok taraflı kurumların temel zayıflığını ortaya koydu.
Kimse şaşırmamalı — bu yıllardır geliyordu. Örüntüler oradaydı: güvenin sessiz erozyonu,
taahhütlerden yavaş çekilme, gerçek bir pozisyonun yokluğunu maskeleyen diplomatik dil.

Çarpıcı olan, krizin kendisi değil, yanıt. Ya da daha doğrusu, yanıtsızlık. Yetkililer
dikkatli konuşuyor, kendilerini hiçbir şeye bağlamayan kelimeler seçiyor. Basın açıklamaları
yapılıyor. Toplantılar planlanıyor. Ve bu sırada, sahada, gerçeklik geri dönülemez biçimde değişiyor.

Orada olan insanlarla konuştum. Yetkililer değil — gerçek insanlar, bunu yaşayanlar.
Anlatımları brifinglerle örtüşmüyor. Hiç örtüşmüyor.
"""

AUTHOR_B_EN = """
Eastern Europe update. Three things happened. Incidents up 40%. NATO issued a statement.
Markets reacted. Let me explain why this matters. 40% is not noise — it's a signal.
NATO's statement: 847 words. Of those, exactly 0 contained any specific commitment.

What to watch: 1) Thursday session. 2) Germany breaking from consensus. 3) Commodity prices.
I track 23 data sources across 7 countries. Daily updates. I don't speculate — I report patterns.
Bottom line: further escalation before stabilization. Timeline: 3-6 weeks.
"""


def test_feature_extraction():
    feat = extract_features(AUTHOR_A_EN)
    assert feat.sent_length_mean > 0
    assert 0 <= feat.lexical_richness <= 1
    assert 0 <= feat.short_sent_ratio <= 1
    assert len(feat.to_vector()) == 26


def test_feature_extraction_turkish():
    feat = extract_features(AUTHOR_A_TR)
    assert feat.sent_length_mean > 0
    assert 0 <= feat.lexical_richness <= 1


def test_same_author_scores_higher(monkeypatch):
    # Without embeddings (fast, no model download in CI)
    result_same = compare(AUTHOR_A_EN, AUTHOR_A_TR, use_embeddings=False)
    result_diff = compare(AUTHOR_A_EN, AUTHOR_B_EN, use_embeddings=False)

    assert result_same.score > result_diff.score, (
        f"Same author ({result_same.score:.1f}) should score higher "
        f"than different author ({result_diff.score:.1f})"
    )


def test_result_fields():
    result = compare(AUTHOR_A_EN, AUTHOR_A_TR, use_embeddings=False)
    assert 0 <= result.score <= 100
    assert result.confidence in ("very low", "low", "medium", "high")
    assert result.verdict
    assert result.text_a_lang == "en"
    assert result.text_b_lang == "tr"
    assert len(result.top_matches) == 5
    assert len(result.top_mismatches) == 5


def test_feature_vector_length():
    feat = extract_features(AUTHOR_B_EN)
    vec = feat.to_vector()
    assert len(vec) == 26, f"Expected 26 features, got {len(vec)}"
