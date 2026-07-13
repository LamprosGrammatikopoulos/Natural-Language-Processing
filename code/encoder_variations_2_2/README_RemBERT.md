# RemBERT Experiments

**Model**: `google/rembert` (~559M parameters)

## Results Summary

### Auto-Selected Configuration
| Setting | Selected | Reason |
|---------|----------|--------|
| Fine-tuning | **LoRA** | F1: 0.9030 vs Full: 0.9015 (within 1%, LoRA more efficient) |
| Balancing | **Weighted** | Best for both tasks |

### Performance by Task

| Task | Pooled F1 | Best Cross-Variety | Worst Cross-Variety |
|------|-----------|-------------------|---------------------|
| **Sentiment** | 0.9074 +/- 0.0052 | en-UK→en-UK (0.9643) | en-AU→en-IN (0.8018) |
| **Sarcasm** | 0.7212 +/- 0.0077 | en-AU→en-AU (0.7528) | en-AU→en-IN (0.4891) |

### Combined Training Results

| Task | Test Variety | Combined F1 | Single-Route Avg | Improvement |
|------|--------------|-------------|------------------|-------------|
| Sentiment | en-AU | 0.9042 | 0.8957 | +0.0085 |
| Sentiment | en-IN | 0.8610 | 0.8325 | +0.0284 |
| Sentiment | en-UK | 0.9642 | 0.9568 | +0.0075 |
| Sarcasm | en-AU | 0.7592 | 0.6361 | +0.1231 |
| Sarcasm | en-IN | 0.6167 | 0.5497 | +0.0670 |
| Sarcasm | en-UK | 0.6878 | 0.6321 | +0.0557 |

## Quick Start

```bash
# Install dependencies
pip install datasets transformers accelerate scikit-learn matplotlib seaborn pandas numpy peft
```

Then open `Q2_2_RemBERT_Experiments.ipynb` and click **Run All**.

## Configuration (Cell 2)

```python
MODEL_NAME = 'google/rembert'
BATCH_SIZE_TRAIN = 8           # Smaller for larger model
BATCH_SIZE_EVAL = 16
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 32
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
LEARNING_RATE_LORA = 2e-4      # Higher LR for LoRA
USE_FP16 = True                # BF16 on Ampere+
SEEDS = [42, 123, 456]
```

## Key Findings

1. **RemBERT outperforms DeBERTa** on both tasks (+6.5% Sentiment, +9.2% Sarcasm)
2. **LoRA matches Full fine-tuning** (0.9030 vs 0.9015) with 98% fewer trainable parameters
3. **Sentiment near-solved**: 90.7% F1 pooled, 96.4% on en-UK
4. **Sarcasm still challenging**: 72.1% F1 pooled
5. **Transfer gap**: Sentiment (1.8%) vs Sarcasm (11.7%)

## Comparison with DeBERTa

| Metric | DeBERTa | RemBERT | Winner |
|--------|---------|---------|--------|
| Sentiment Pooled F1 | 0.8425 | **0.9074** | RemBERT (+6.5%) |
| Sarcasm Pooled F1 | 0.6290 | **0.7212** | RemBERT (+9.2%) |
| Best Sentiment Route | 0.8402 | **0.9643** | RemBERT |
| Best Sarcasm Route | 0.5972 | **0.7528** | RemBERT |
| Training Time | ~90s/run | ~450s/run | DeBERTa (5x faster) |
| LoRA Trainable % | 0.32% | 1.82% | DeBERTa (more efficient) |

## Outputs

```
outputs_rembert/
├── sentiment/
│   ├── pooled_results.csv
│   ├── cross_variety_results.csv
│   ├── combined_results.csv
│   └── final_models/
└── sarcasm/
    └── ...
```

## Runtime

| Config | Time (A4000) |
|--------|--------------|
| Full (3 seeds, both tasks) | ~7-8 hours |
| Quick (`SEEDS=[42]`) | ~2 hours |

## Troubleshooting

### CUDA Out of Memory
```python
BATCH_SIZE_TRAIN = 4
GRADIENT_ACCUMULATION_STEPS = 8
```

### Slow Training
RemBERT is larger than DeBERTa - LoRA is auto-selected for efficiency.

See main [README.md](README.md) for detailed documentation.
