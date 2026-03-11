# idiolect

**Cross-lingual author attribution — find the same voice across different languages.**

```
idiolect compare author_en.txt suspect_tr.txt
```

```
╭─────────────────────────────────────────────╮
│ idiolect — Cross-lingual Author Attribution │
╰─────────────────────────────────────────────╯

  File A: author_en.txt  (detected: en)
  File B: suspect_tr.txt (detected: tr)

  Attribution Score
  [████████████████████████████████░░░░░░░░] 73.3/100

  Verdict:     Strong evidence of common authorship
  Confidence:  low

  Structural features              61.7%
  Semantic style (multilingual)    94.8%
```

---

## What is idiolect?

An **idiolect** is the specific variety of a language used by one person — their unique fingerprint in how they write. People carry their idiolect across languages: if someone writes in English *and* Turkish *and* Russian, they bring the same sentence rhythm, punctuation habits, and structural patterns to all three.

`idiolect` detects this cross-lingual fingerprint.

It was built for:
- OSINT investigators tracking pseudonymous actors across platforms
- Disinformation researchers identifying coordinated inauthentic behavior
- Journalists verifying source identity
- Forensic linguists studying influence operations

---

## How it works

Two parallel analysis pipelines:

### 1. Structural feature extraction (language-agnostic)
Extracts 26 writing habit features that persist across languages:

| Category | Features |
|---|---|
| Sentence rhythm | mean/std/min/max sentence length, short/long sentence ratio |
| Punctuation habits | comma rhythm, question/exclamation ratio, ellipsis, dash, colon, semicolon density |
| Word-level | average word length, lexical richness (normalized TTR), word length distribution |
| Structural | paragraph density, average paragraph length |
| Behavioral | mid-sentence capitalization, number usage density |

### 2. Multilingual style embeddings
Uses [`paraphrase-multilingual-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) — a transformer model supporting 50+ languages that maps text into a shared semantic space regardless of language.

Texts are chunked, embedded, then averaged into a single style vector. Cosine similarity between vectors captures deep stylistic patterns that structural features miss.

### Fusion
Cross-lingual comparisons weight structural features more heavily (65%) since they are topic-independent. Same-language comparisons weight both equally (50/50).

---

## Installation

```bash
pip install idiolect
```

Or from source:
```bash
git clone https://github.com/sa-aris/idiolect
cd idiolect
pip install -r requirements.txt
```

First run downloads the multilingual model (~420MB, cached after first use).

---

## Usage

### Compare two texts
```bash
idiolect compare text_a.txt text_b.txt
```

### Skip ML model (structural features only, faster)
```bash
idiolect compare text_a.txt text_b.txt --no-embeddings
```

### JSON output (for scripting and pipelines)
```bash
idiolect compare text_a.txt text_b.txt --output json
idiolect multi known.txt s1.txt s2.txt --output json
idiolect batch ./corpus/ --output json
```

### One-against-many: rank suspects by similarity to a known author
```bash
idiolect multi known_author.txt suspect1.txt suspect2.txt suspect3.txt
```

Output is ranked by score, highest first.

### Compare all files in a directory (corpus mode)
```bash
idiolect batch ./corpus/ --threshold 65
```

Compares every `.txt` file against every other and lists pairs above the score threshold. Useful for finding clusters of related texts in a large corpus.

### Inspect style features of a single text
```bash
idiolect features text.txt
```

---

## Interpreting scores

| Score | Verdict |
|---|---|
| 72–100 | Strong evidence of common authorship |
| 60–71 | Moderate evidence of common authorship |
| 46–59 | Inconclusive |
| 32–45 | Moderate evidence of different authors |
| 0–31 | Strong evidence of different authors |

**Confidence** depends on text length:

| Minimum text length | Confidence |
|---|---|
| 500+ words | high |
| 200–499 words | medium |
| 80–199 words | low |
| < 80 words | very low |

For reliable results, use at least **200 words per sample**. Short texts reduce accuracy significantly.

---

## Limitations (be honest with yourself)

- **Topic contamination**: When comparing texts on the *same topic* in the *same language*, semantic embeddings may pick up shared vocabulary rather than shared style. Mitigate by using samples from different topics.
- **Translation artifacts**: Machine-translated text does not preserve the author's idiolect — it reflects the translator model's style instead.
- **Genre differences**: Comparing a blog post against an academic paper written by the same person will score lower than expected — genre forces style changes.
- **Text length**: Below 150 words, results are unreliable. More text = more signal.
- **v0.1 note**: This is an early release. The model has not been trained on labeled authorship data — it uses general-purpose multilingual embeddings. Accuracy will improve with fine-tuning on authorship corpora.

---

## Real-world use case

A disinformation research team suspects that an anonymous English-language Twitter account and a Russian-language Telegram channel are operated by the same person. They export 500 words from each and run:

```bash
idiolect compare twitter_export.txt telegram_export.txt
```

The tool returns a score and a breakdown of which stylistic features match and which diverge — giving investigators a starting point for deeper analysis, not a final verdict.

**idiolect is a hypothesis generator, not a court of law.**

---

## Python API

```python
from idiolect import compare

result = compare(text_english, text_turkish)

print(result.score)           # 0-100
print(result.verdict)         # human-readable conclusion
print(result.confidence)      # "low" | "medium" | "high"
print(result.feature_score)   # structural similarity 0-1
print(result.embedding_score) # semantic similarity 0-1
print(result.top_matches)     # [(feature_name, val_a, val_b, similarity), ...]
print(result.top_mismatches)  # diverging features
```

Skip embeddings for lightweight integration:
```python
result = compare(text_a, text_b, use_embeddings=False)
```

---

## Roadmap

- [ ] Fine-tune on PAN authorship verification datasets
- [ ] Per-language calibration (structural features mean different things in agglutinative vs analytic languages)
- [ ] Handle code-switching (texts that mix languages)
- [ ] Confidence intervals via bootstrap sampling
- [ ] REST API server mode
- [ ] Support for social media input (direct username → fetch → compare)

---

## Background

Most authorship attribution tools are monolingual. This matters because real-world influence operations are not. The same operator often writes in multiple languages simultaneously — and while they may adapt their vocabulary and grammar, their deep writing habits persist.

The core hypothesis: **an author's idiolect is partially language-invariant.** Sentence rhythm, punctuation density, paragraph structure, and lexical richness are partially determined by cognitive and behavioral habits that transcend the language being used.

This tool operationalizes that hypothesis.

---

## License

MIT

---

## Contributing

Issues and PRs welcome. Especially useful:
- Labeled cross-lingual authorship datasets for evaluation
- Validation against known influence operation cases (Ghostwriter, Secondary Infektion, etc.)
- Language-specific feature calibration
