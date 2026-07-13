# Encoder Variations: DeBERTa & RemBERT Experiments

This directory contains comprehensive experiments comparing **DeBERTa-v3-base** and **RemBERT** encoder models for sentiment analysis and sarcasm detection across English varieties (en-AU, en-IN, en-UK).

## Results Summary

### Model Comparison

| Metric | DeBERTa | RemBERT | Winner |
|--------|---------|---------|--------|
| **Sentiment Pooled F1** | 0.8425 +/- 0.0049 | **0.9074 +/- 0.0052** | RemBERT (+6.5%) |
| **Sarcasm Pooled F1** | 0.6290 +/- 0.0167 | **0.7212 +/- 0.0077** | RemBERT (+9.2%) |
| Best Sentiment Route | en-UK→en-UK (0.8402) | **en-UK→en-UK (0.9643)** | RemBERT |
| Best Sarcasm Route | en-AU→en-AU (0.5972) | **en-AU→en-AU (0.7528)** | RemBERT |
| Training Time (per run) | **~90s** | ~450s | DeBERTa (5x faster) |
| LoRA Trainable Params | **0.32%** | 1.82% | DeBERTa |

### Auto-Selected Configurations

| Model | Fine-tuning | Balancing | Reason |
|-------|------------|-----------|--------|
| DeBERTa | **LoRA** | Weighted | Full fine-tuning unstable (F1 collapsed to 0.34) |
| RemBERT | **LoRA** | Weighted | LoRA matches Full (0.903 vs 0.902), more efficient |

### Cross-Variety Transfer Matrix

**Sentiment (RemBERT)**
| Train → Test | en-AU | en-IN | en-UK |
|--------------|-------|-------|-------|
| en-AU | 0.8991 | 0.8018 | 0.9226 |
| en-IN | 0.8876 | 0.8346 | 0.9571 |
| en-UK | 0.9001 | 0.8613 | **0.9643** |

**Sarcasm (RemBERT)**
| Train → Test | en-AU | en-IN | en-UK |
|--------------|-------|-------|-------|
| en-AU | **0.7528** | 0.4891 | 0.6661 |
| en-IN | 0.6113 | 0.5986 | 0.6494 |
| en-UK | 0.5440 | 0.5615 | 0.7001 |

## Overview

| Notebook | Model | Parameters | Recommended GPU |
|----------|-------|------------|-----------------|
| `Q2_2_DeBERTa_Experiments.ipynb` | microsoft/deberta-v3-base | ~440M | A4000 (16GB) |
| `Q2_2_RemBERT_Experiments.ipynb` | google/rembert | ~559M | A4000 (16GB) |

## Key Findings

1. **RemBERT outperforms DeBERTa** on both tasks by significant margins
2. **LoRA is the recommended fine-tuning method** for both models
3. **Sentiment is near-solved**: RemBERT achieves 90.7% F1 pooled, 96.4% on en-UK
4. **Sarcasm remains challenging**: Best is 72.1% F1 (RemBERT pooled)
5. **en-IN is the hardest variety** for both models and tasks
6. **Combined training helps Sarcasm more than Sentiment**: +12% vs +2% average improvement
7. **Transfer gap is larger for Sarcasm** (11.7% RemBERT) vs Sentiment (1.8%)

## Experiments Included

### 1. Dataset Balancing Comparison
Compares strategies to handle class imbalance:
- **Weighted Loss** - Class-weighted cross-entropy (selected for both models)
- **Random Oversampling** - Duplicate minority samples
- **Random Undersampling** - Reduce majority samples

### 2. Fine-tuning Method Comparison
- **Full Fine-tuning** - Update all parameters (100%)
- **LoRA (PEFT)** - Parameter-efficient fine-tuning (~0.3-1.8% parameters)

### 3. Training Configurations

| Experiment | Description |
|------------|-------------|
| **Pooled** | Train on all varieties together, test on combined test set |
| **Cross-variety** | Train on one variety, test on all varieties (transfer matrix) |
| **Combined** | Train on all varieties, evaluate separately by target variety |

### 4. Dual-Task Evaluation
- **Sentiment Analysis** (positive/negative)
- **Sarcasm Detection** (sarcastic/non-sarcastic)

## Quick Start

### 1. Install Dependencies
```bash
pip install datasets transformers accelerate scikit-learn matplotlib seaborn pandas numpy peft
```

### 2. Run Experiments
Open the notebook and click **"Run All"** - configs are pre-optimized for A4000 GPU.

### 3. Configuration (Optional)
Edit **Cell 2** to customize:

```python
# Speed up initial testing
SEEDS = [42]              # Default: [42, 123, 456]
NUM_EPOCHS = 3            # Default: 5

# Resume behavior
RESUME_FROM_CHECKPOINT = True  # Skip completed experiments on re-run
```

## GPU Optimization (A4000 16GB)

| Setting | DeBERTa | RemBERT |
|---------|---------|---------|
| Batch Size (train) | 16 | 8 |
| Batch Size (eval) | 32 | 16 |
| Gradient Accumulation | 2 | 4 |
| Effective Batch | 32 | 32 |
| FP16/BF16 | Disabled* | Enabled |
| Learning Rate | 2e-5 | 2e-5 |
| LoRA Learning Rate | 2e-4 | 2e-4 |

*DeBERTa has training instability with mixed precision

## Resume on Interruption

If training stops (PC shutdown, etc.), simply **run the notebook again**:
- Completed experiments are automatically skipped
- Results are saved after each seed
- Progress is shown: `SKIPPING seed=42 (already completed)`

To force re-run everything:
```python
RESUME_FROM_CHECKPOINT = False
```

## Output Files

### Results (CSV)
| File | Description |
|------|-------------|
| `pooled_results.csv` | Pooled training metrics by seed |
| `cross_variety_results.csv` | Cross-variety transfer matrix |
| `combined_results.csv` | Combined training by target variety |
| `balancing_comparison_seed42.csv` | Balancing strategy comparison |
| `finetuning_method_comparison.csv` | LoRA vs Full comparison |

### Plots (PNG)
| File | Description |
|------|-------------|
| `balancing_comparison_*.png` | Balancing strategy results |
| `finetuning_method_comparison.png` | LoRA vs Full fine-tuning |
| `pooled_training_history_*.png` | Loss/F1 curves |
| `pooled_confusion_matrix_*.png` | Confusion matrices |
| `cross_variety_*_heatmap.png` | Transfer matrix heatmaps |
| `combined_training_metrics_comparison.png` | Combined results |

### Models (Saved Checkpoints)
```
outputs_deberta/          outputs_rembert/
├── sentiment/            ├── sentiment/
│   ├── checkpoints/      │   ├── checkpoints/
│   └── final_models/     │   └── final_models/
└── sarcasm/              └── sarcasm/
```

## Expected Runtime

| Model | Full Run (3 seeds) | Quick Run (1 seed) |
|-------|-------------------|-------------------|
| DeBERTa | ~6-7 hours | ~1.5 hours |
| RemBERT | ~7-8 hours | ~2 hours |

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in Cell 2:
```python
BATCH_SIZE_TRAIN = 8   # DeBERTa
BATCH_SIZE_TRAIN = 4   # RemBERT
```

### NaN Loss (DeBERTa)
DeBERTa can be unstable with mixed precision:
```python
USE_FP16 = False
USE_BF16 = False
```

### Missing Dependencies
```bash
pip install peft
```

## References

- [DeBERTa-v3](https://huggingface.co/microsoft/deberta-v3-base)
- [RemBERT](https://huggingface.co/google/rembert)
- [BESSTIE Dataset](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26)
- [PEFT/LoRA](https://huggingface.co/docs/peft)

## Individual Model READMEs

- [DeBERTa Results](README_DeBERTa.md)
- [RemBERT Results](README_RemBERT.md)
