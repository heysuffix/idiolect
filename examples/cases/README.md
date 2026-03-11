# Case Study: Cross-lingual Stylometric Attribution

## Scenario

This case study demonstrates idiolect on a synthetic but realistic scenario
modeled on the **Ghostwriter** influence operation (documented by Mandiant, 2020-2021).

Ghostwriter was a Russian-linked operation that planted fabricated content across
Polish, Lithuanian, German, and Latvian media. Key characteristic: the same operators
wrote in multiple languages, and their writing style was consistent across languages.

## Files

| File | Description |
|---|---|
| `ghostwriter_pl.txt` | Polish text — same operator, same stylistic fingerprint |
| `ghostwriter_de.txt` | German text — same operator, translated style |
| `control_journalist_en.txt` | Control: genuine journalist, different writing style |

**Important:** These texts are synthetic demonstrations created to illustrate
the stylometric patterns. They are not actual Ghostwriter operation content.
The structural patterns (sentence rhythm, punctuation style, rhetorical structure)
are modeled on the publicly documented characteristics of the operation.

## Running the demo

```bash
# Same operator: Polish vs German (expect high score)
idiolect compare ghostwriter_pl.txt ghostwriter_de.txt

# Different operator: Ghostwriter vs control journalist (expect lower score)
idiolect compare ghostwriter_pl.txt control_journalist_en.txt

# Bootstrap CI for more rigorous estimate
idiolect compare ghostwriter_pl.txt ghostwriter_de.txt --bootstrap

# Full matrix
idiolect matrix ghostwriter_pl.txt ghostwriter_de.txt control_journalist_en.txt --heatmap matrix.png
```

## Expected results

| Comparison | Expected score | Reasoning |
|---|---|---|
| PL vs DE (same op) | 68–78 | Same rhetorical patterns, same structural habits |
| PL vs EN control | 40–55 | Different style — journalist uses data-heavy, impersonal tone |
| DE vs EN control | 40–55 | Same divergence |

## What to look for

The Ghostwriter operator's distinguishing features:
- **Rhetorical escalation pattern**: starts with "X must decide", builds to "someone chose this"
- **Embedded sourcing**: "I spoke with people who were there. Not officials — real people."
- **Closing aphorism**: ends with a short, standalone philosophical sentence
- **Punctuation rhythm**: heavy dash use, consistent paragraph length (4-5 sentences)
- **Question as accusation**: "Who benefits from this?" used as rhetorical pivot

The control journalist's distinguishing features:
- Data-anchored writing (numbers, percentages, dates)
- Short sentences with minimal rhetorical flourish
- Attribution-heavy ("Officials I spoke with...")
- No rhetorical questions

## Further reading

- Mandiant: "UNC1151 Assessed with High Confidence to Have Links to Belarus" (2021)
- EU DisinfoLab: "Ghostwriter" campaign documentation
- DFRLab: Analysis of Baltic state influence operations
