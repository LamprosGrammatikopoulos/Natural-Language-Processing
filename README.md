# Dialect-Aware Sentiment & Sarcasm Classification for Varieties of English

**COMM061 Natural Language Processing — Group Coursework (Group 5)**
MSc Artificial Intelligence, University of Surrey, 2025–26

Supervisors: Dr. D. Kanojia, Dr. S. De, Dr. B. Vrusias

---

## Overview

NLP systems trained on standard, inner-circle English degrade when confronted with dialectal variation. This is especially damaging for **pragmatic** tasks such as sarcasm detection, where the surface form of an utterance is deliberately incongruent with its intended meaning, and where that incongruence is realised through variety-specific conventions (Australian hyperbolic understatement, British dry wit, Hindi–English code-mixing in Indian English).

This project quantifies and attacks that **variety gap** on the [BESSTIE](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26) benchmark, which spans **British (en-UK)**, **Australian (en-AU)** and **Indian (en-IN)** English across Google Places reviews and Reddit comments, with binary labels for **Sentiment** (Positive/Negative) and **Sarcasm** (Sarcastic/Not Sarcastic).

The work covers the full pipeline: dataset and vocabulary analysis, three controlled experimental setups (classical vs. PTLM, cross-variety encoders, LoRA-adapted LLMs), per-class evaluation, few-shot explanation-based error recovery, and a deployed Gradio inference endpoint with an efficiency analysis.

---

## Dataset

| Split | Rows | en-AU | en-IN | en-UK | Google | Reddit | Sarcastic | Not Sarcastic |
|---|---|---|---|---|---|---|---|---|
| Train | 3,747 | 1,145 | 1,399 | 1,203 | 1,874 | 1,873 | 524 | 3,223 |
| Validation | 313 | 95 | 117 | 101 | 167 | 146 | 44 | 269 |
| Test | 2,183 | 667 | 816 | 700 | 1,101 | 1,082 | 305 | 1,878 |

Sentiment is near-balanced across all three varieties. Sarcasm is severely imbalanced — only **6.8%** of en-IN and **7.6%** of en-UK instances are sarcastic — so **Macro-F1** is the primary metric throughout, and a naive majority-class predictor is treated as a failure regardless of accuracy.

---

## What the report implements

### 1. Dataset analysis (visualisation + vocabulary)
- Label distributions per variety and per split; domain effects (sarcasm is essentially absent from Google reviews — 0.1% for en-UK — and concentrated in Reddit, peaking at 41.7% for en-AU).
- **POS-tag distribution** across varieties, showing lower determiner frequency and higher noun density in en-IN, consistent with documented Indian English grammar (article omission) rather than mere topical drift.
- **Vocabulary overlap** via Jaccard similarity (en-AU/en-UK = 0.2576; both en-IN pairings lower) and TF-IDF cosine similarity (en-AU/en-UK = 0.8649; en-AU/en-IN = 0.7357), plus a written definition of **linguistic distance** and an argument that the en-IN gap is partly structural, not purely topical.
- Detection of **Hindi–English code-mixing** ("hai", "toh", "kya") concentrated in sarcastic en-IN text.
- **Synthetic data comparison**: 1,244 LLM-generated BESSTIE-style instances were produced and compared against a size-matched real sample. The synthetic corpus flattens the variety gap (higher within-variety Jaccard), treats sarcasm as a coin flip (50.6% vs. 14.7% real), and misses 9,429 real vocabulary tokens — so it was **rejected** for augmentation.
- **Class-imbalance ablation** (TF-IDF + Logistic Regression, 3 seeds): no balancing collapses to Sarcastic-F1 = 0.0000 despite 86% accuracy; class weighting reaches Macro-F1 0.6275, oversampling 0.6156. Class weighting is adopted downstream where supported.

### 2.1 Baseline / PTLM gap
TF-IDF + Logistic Regression, TF-IDF + LinearSVC, FastText (supervised), and fine-tuned RoBERTa-base, pooled across varieties, 3 seeds each.

The pre-training gap is roughly **twice as large in relative terms on the pragmatic Sarcasm task (+11.6%) than on the lexical Sentiment task (+6.0%)**. Notably, the RoBERTa advantage on sarcasm **disappears on en-IN** (−0.004 vs. the classical baseline) while remaining strong on the inner-circle varieties — an Outer-Circle reversal.

### 2.2 Cross-variety encoder evaluation
DeBERTa-v3-base, RemBERT and XLM-RoBERTa-base, fine-tuned under three training modes (pooled, per-variety, and a full **3×3 cross-variety transfer matrix**), 3 seeds each.

RemBERT wins throughout (pooled Macro-F1 0.9074 Sentiment / 0.7212 Sarcasm). Sentiment transfers well across varieties; sarcasm does not — inner-circle-trained models collapse to 0.49–0.56 Macro-F1 on en-IN, exposing the variety gap directly. Per-class metrics confirm majority-class bias (Sarcastic precision as low as 0.145 for DeBERTa on en-IN).

### 2.3 LLM variations with LoRA adapters ✦ *(my section)*
Four architectures compared on the Sarcasm task across all three varieties plus a pooled adapter:

| Architecture | Type | Params | Adaptation |
|---|---|---|---|
| TinyLlama-1.1B | Decoder (English-centric) | 1.1B | LoRA (q/v proj) |
| Qwen2.5-1.5B | Decoder (multilingual, 29 langs) | 1.5B | LoRA (q/v proj) |
| XLM-RoBERTa-base | Encoder (multilingual, 100 langs) | 278M | Full fine-tuning |
| XLM-RoBERTa-base | Encoder | 278M | LoRA (query/value) |

The dual XLM-R inclusion is deliberate: holding the backbone, tokeniser, data, seeds and schedule constant while varying **only** the adaptation strategy yields a **controlled LoRA-vs-full-FT ablation**, isolating parameter-efficiency from backbone family.

**Key findings:**
- **Perplexity as a pre-training probe.** Frozen-backbone perplexity ranks en-UK < en-AU < en-IN for *both* LLMs, showing en-IN is genuinely out-of-distribution — harder to *model*, not just harder to classify. TinyLlama shows 38–41% lower perplexity than Qwen2.5 despite being 27% smaller, and this predicts the downstream ordering.
- **LoRA beats full fine-tuning on the same backbone in every in-distribution cell** (+0.05–0.07 Macro-F1, +0.04–0.07 Sarcasm-F1), with a **7.2× reduction in run-to-run variance** on en-AU (spread 0.0162 vs. 0.1167) — the low-rank constraint acting as an implicit regulariser in the small-data regime.
- **Negative variety gap on en-IN** for both XLM-R variants (−0.0743 full FT, −0.0056 LoRA): the en-IN-trained encoder transfers *as well or better* to other varieties than it performs in-distribution. This reversal is absent in both decoders, suggesting bidirectional representations capture more transferable pragmatic features.
- **Pooling is not universally beneficial.** Decoders prefer the combined adapter on en-IN (where the 800-sample budget × ~13% sarcasm prevalence is too thin); the encoder prefers per-variety on en-IN and pooling on en-AU.
- **Competitive with far larger models.** XLM-R-base + LoRA reaches 0.6433 Sarcasm-F1 on en-AU, surpassing the 0.62 reported for XLM-R-*Large* full fine-tuning in Srirag et al., at half the size.

Zero-shot baselines are reported for every cell, and the two opposite collapse modes observed (all-sarcastic vs. all-non-sarcastic, purely from classification-head seed) are used to argue that randomly-headed zero-shot baselines are dominated by initialisation luck.

### 4. Sarcasm explanation & few-shot error analysis
Ten in-distribution errors were sampled from the TinyLlama LoRA adapters; four were turned into a **metacognitive three-step explanation prompt** (surface polarity → real situation → meaning flip), ordered to counter recency bias. The remaining six were tested three ways: LoRA adapter (wrong by construction), zero-shot base model (3/6 recovered), and four-shot explained prompt (**4/6 recovered**). Zero-shot recovers all False Negatives only because it predicts "Sarcastic" for everything; four-shot preserves that recall *and* recovers a False Positive through genuine chain-of-thought revision.

### 5. Deployment endpoint & efficiency
A **Gradio** web service hosting every trained model, organised into tabs by experiment (baselines / cross-variety encoders / LLM variations). The user selects an architecture, a variety, and enters text; the backend returns the label, a confidence score, and inference latency, with a flagging button that writes mispredictions to per-model CSVs for future retraining.

Routing uses an `ADAPTER_REGISTRY` / `MODEL_REGISTRY` design — a single dictionary entry maps each `(architecture, variety)` pair to its checkpoint or adapter path, so adding a variety requires no new inference code. The **LoRA hot-swap pattern** is the architectural basis: the base model is loaded into GPU memory once and only the adapter weights are swapped per request.

Efficiency analysis covers standard and large-batch inference across all three model families, showing a clear three-tier structure (classical ≈0.09–0.44 ms/sample → encoders ≈0.9–10.4 ms → decoder LLMs 8.2–11.5 ms), plus the on-disk storage trade-off (~177 MB for three TinyLlama adapters vs. ~1.68 GB for three full-FT XLM-R checkpoints).

---

## My contributions (Lampros Grammatikopoulos)

- **Full ownership of Sections 2.3 and 3.3** — design, implementation, experimentation and write-up of the LoRA/LLM track: the four-architecture comparison, the controlled LoRA-vs-full-FT ablation on a fixed XLM-R backbone, the perplexity/pre-training calibration study, the 4×4 cross-variety transfer matrices, the variety-gap metric and the negative-gap finding, the adaptation-gain and pooling analysis, and all associated training/evaluation code.
- **Codebase integration** — merged the separate per-member experiment notebooks into a coherent, runnable pipeline with consistent splits, seeding, metrics and result schemas.
- **Deployment (5.1)** — provided the ready-made callable inference scripts and model/adapter loading interfaces that the Gradio endpoint is built on (registry-based routing, unified `load_model_and_predict()` wrappers over heterogeneous model types).
- **Efficiency (5.2)** — contributed the latency-measurement code (CUDA-synchronised, batched, per-variety).
- **Report** — wrote and formatted the full report end to end, including sections authored by other members, and produced the final structure, tables and figure captions.

---

## Repository structure

```
.
├── main.ipynb                 # Main notebook — all experiments, delineated by section headers
├── notebooks/                 # Per-experiment notebooks (1, 2.1, 2.2, 2.3, 4)
├── src/
│   ├── registry.py            # ADAPTER_REGISTRY / MODEL_REGISTRY
│   ├── inference.py           # load_model_and_predict() wrappers
│   └── latency.py             # CUDA-synchronised timing utilities
├── app/
│   └── app.py                 # Gradio deployment endpoint
├── figures/                   # All report figures
├── requirements.txt
└── README.md
```

> Trained checkpoints, LoRA adapters and the dataset are **not** committed. The dataset is pulled from the Hugging Face Hub; adapters can be hosted on the Hub and loaded via `from_pretrained()`.

---

## Reproducing

```bash
pip install -r requirements.txt
jupyter lab main.ipynb          # experiments
python app/app.py               # Gradio endpoint
```

All experiments use fixed seeds (42/123/456 for Sections 2.1–2.2; 43/44/45 for Section 2.3) and report mean ± standard deviation over at least three runs. The test split is never modified or seen during training or model selection.

---

## References

Key works underpinning the project: Srirag et al. (BESSTIE, ACL Findings 2025); Kachru (Inner/Outer Circle, 1985); Hu et al. (LoRA, ICLR 2022); Conneau et al. (XLM-R, ACL 2020); Chung et al. (RemBERT, ICLR 2021); He et al. (DeBERTaV3, ICLR 2023); Abercrombie & Hovy (sarcasm class imbalance, 2016); Singh, Srirag & Joshi (Nek Minit, 2025); Plank (human label variation, EMNLP 2022). Full reference list in the report.

---

## Team

Lampros Grammatikopoulos · Nikolas Ziartis · Thant Zaw Latt · Mohd Amir Suhel · Alexander Wallop
