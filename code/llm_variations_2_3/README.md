# Section 2.3 - LLM Variations with LoRA Adapters
**Member 4: Lampros Grammatikopoulos (URN: 6918674) | Group 5 | NLP COMM061, University of Surrey, 2025-26**

---

## Overview

This notebook implements **Section 2.3** of the coursework: training lightweight
**LoRA (Low-Rank Adaptation)** adapters on a frozen open-weight LLM for sarcasm
detection across three English varieties from the BESSTIE dataset. It has also been extended (Sections D, E, F, G) to include an encoder-based model (XLM-RoBERTa), a multilingual LLM (Qwen2.5-1.5B), and a detailed perplexity-based pre-training analysis.

- **Dataset Caching**: Implemented disk-based caching (`cached_tokenized_datasets_*`) to accelerate training start-up.
- **Standardized Metrics**: Evaluation now logs per-class **Precision, Recall, and F1** (focussing on the *Sarcastic* class) to prevent majority-class bias.
- **Unified CSV Schema**: Optimized `tinyllama_training_runs.csv` to track `best_val_macro_f1` and training `time_s`, including a `selected` run indicator for cross-variety evaluation.
- **Organized Results**: Automated grouping of experiment outputs (CSVs, matrices, and reports) into a logical `./results/tinyllama/` structure.

**Base Models**: 
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~1.1B parameters, decoder, English-dominant)
- `xlm-roberta-base` (~270M parameters, encoder, multilingual)
- `Qwen/Qwen2.5-1.5B` (~1.5B parameters, decoder, multilingual)

| Adapter | Trained on | Base Model |
|---------|------------|------------|
| UK Adapter | en-UK (British English) | Base Model |
| AU Adapter | en-AU (Australian English) | Base Model |
| IN Adapter | en-IN (Indian English) | Base Model |
| Combined Adapter | en-UK + en-AU + en-IN (pooled) | Base Model |

Each architecture is evaluated using a **4×4 error rate analysis / heatmaps** across all language varieties (UK, AU, IN, and Combined), covering both in-distribution and zero-shot transferability scenarios.

---

## Files

```
lora_adapters_full.ipynb         ← Main notebook (original Section 2.3 + Extensions A–G)
inference_lora_datapters_full.py ← CLI inference script (multi-architecture support)
requirements.txt                 ← Python dependencies
README.md                        ← This file
```

### Notebook Structure

| Section | Content |
|---------|---------|
| 0 - Environment Setup | Package installation, imports, GPU config |
| 1 - Imports & Configuration | CONFIG dict, reproducibility seeds |
| 2 - Dataset Loading | BESSTIE dataset from HuggingFace |
| 3 - Data Preparation | Class imbalance handling via oversampling |
| 4 - Tokenisation | Multi-model tokenization with **disk-based caching** |
| 5 - Base Model & LoRA Config | LoRA rank/alpha/target modules |
| 6 - Training | TinyLlama LoRA Training (3 runs × 3+1 varieties) |
| 7 - Evaluation | TinyLlama 4x4 inference matrix (Standardized Metrics) |
| 8 - Results & Visualisation | Tables (Sarcasm F1 focus), heatmaps, confusion matrices |
| 9 - Efficiency & Summary | ms/sample, adapter size vs full model |
| **A - Combined Adapter** | Pooled adapter trained on all 3 varieties |
| **B - Zero-Shot Baseline** | Frozen TinyLlama with no adapter (lower bound) |
| **C - Error Export** | Per-sample mispredictions CSV for Q4 analysis |
| **D - XLM-RoBERTa** | Full fine-tuning of 270M encoder baseline |
| **E - Qwen2.5-1.5B** | LoRA adapters for multilingual decoder LLM (16-cell matrix) |
| **F - Architecture Comparison**| Side-by-side: TinyLlama vs XLM-R vs Qwen2.5 |
| **G - Perplexity Analysis** | Causal perplexity vs LoRA gain (pre-training calibration) |

---

## Outputs

**Results folder** (`results/`):
```
results/
|-- data/                            ← Dataset distributions
|-- tinyllama/                       ← TinyLlama specific heatmaps and CSVs
|-- qwen25/                          ← Qwen2.5 specific heatmaps and CSVs
|-- xlmr/                            ← XLM-RoBERTa specific heatmaps and CSVs
|-- comparison/                      ← Cross-architecture comparison results
│   |-- architecture_comparison.png  ← Multi-model F1 bar charts
│   |-- architecture_comparison.csv  ← Consolidated F1 scores
│   |-- perplexity_comparison.csv    ← Causal perplexity per variety
│   |-- perplexity_vs_lora_gain.png  ← Correlation between PPL and adaptation boost
│   |-- transfer_to_en_in.csv        ← Performance on Indian English variety
|-- error_analysis.csv               ← Combined mispredictions for Q4 analysis
```

**Adapter and Checkpoint folders**:
- `tinyllama_adapters/` - LoRA weights for TinyLlama (~4 MB each)
- `qwen_adapters/` - LoRA weights for Qwen2.5 (~8 MB each)
- `xlmr_checkpoints/` - Full model weights for XLM-R (~1.1 GB each - **excluded from ZIP**)

Each `run*/` subfolder contains:
- `adapter_model.safetensors` - LoRA weights only
- `adapter_config.json` - LoRA configuration
- `train_log.json` - per-epoch validation metrics

---

## Setup

### Prerequisites
- Python **3.14**
- CUDA-capable GPU (recommended: ≥12 GB VRAM for Qwen2.5)
- Internet access to download model and dataset from HuggingFace

### Install dependencies

```bash
# 1. Install Blackwell-native PyTorch (CUDA 12.8 nightly)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 2. Fix Python 3.14 multiprocessing and datasets conflict
pip install multiprocess>=0.70.19 datasets>=4.6.1 --no-deps

# 3. Install remaining dependencies
pip install -r requirements.txt
```

### Launch the notebook

```bash
jupyter notebook lora_adapters_full.ipynb
```

---

## Inference Script

`inference_lora_datapters_full.py` allows testing any trained adapter or fine-tuned checkpoint across all three architectures.

### Usage

```bash
# Evaluate ALL three architectures and perform a majority vote!
python inference_lora_datapters_full.py --arch all --variety en-UK --text "Oh brilliant, it's raining again."

# Test Qwen2.5 on the combined adapter
python inference_lora_datapters_full.py --arch qwen25 --variety combined --text "Absolute legend."
```

---

## What this notebook produces (for the report)

| Rubric Item | Produced by |
|-------------|-------------|
| LoRA adapter for each variety (UK/AU/IN) | Section 6 & E - training loops |
| Combined (pooled) adapter | Section A & E - combined training |
| Zero-shot lower bound | Section B & E.12 - zero-shot baselines |
| 4x4 Error rate analysis / heatmaps | Section 8 & E.18 - performance matrices |
| Multilingual Comparison | Section F - TinyLlama vs Qwen2.5 vs XLM-R |
| Perplexity Correlation | Section G - PPL vs LoRA gain analysis |
| Error export for Q4 analysis | Section C & E.14 - `error_analysis.csv` |
| Efficiency data for Q5.2 | Section 9 & F.4 - `efficiency_summary.csv` |

---

To integrate into the group's `main.ipynb`:

1. Copy Sections 2–9 and A–C into `main.ipynb` under a `## 2.3 LLM Variations` heading.
2. The `adapter_registry` dict can be passed directly to Member 5's deployment code:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, adapter_registry["en-UK"][0])
```

3. For Q4 error analysis, Member 2 can load errors directly:

```python
import pandas as pd
errors = pd.read_csv("results/{model}/{model}_error_analysis.csv")
# In-distribution errors only (most useful for Q4)
indist_errors = errors[errors["adapter_variety"] == errors["test_variety"]]
print(indist_errors.head(10))
```

---

## References

- Hu, E. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Srirag et al. (2025). *BESSTIE: A Benchmark for Sentiment and Sarcasm Classification for Varieties of English.* ACL 2025 Findings.
- Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020.
- HuggingFace PEFT Library: https://github.com/huggingface/peft
- TinyLlama: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Qwen2.5: https://huggingface.co/Qwen/Qwen2.5-1.5B
- XLM-RoBERTa (xlmr): https://huggingface.co/xlm-roberta-base

