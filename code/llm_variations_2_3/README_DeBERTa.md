# DeBERTa Experiments

**Model**: `microsoft/deberta-v3-base` (~440M parameters)

## Results Summary

### Auto-Selected Configuration
| Setting | Selected | Reason |
|---------|----------|--------|
| Fine-tuning | **LoRA** | F1: 0.8524 vs Full: 0.3385 (Full had training instability) |
| Balancing | **Weighted** | Best for both tasks |

### Performance by Task

| Task | Pooled F1 | Best Cross-Variety | Worst Cross-Variety |
|------|-----------|-------------------|---------------------|
| **Sentiment** | 0.8425 +/- 0.0049 | en-UK→en-UK (0.8402) | en-IN→en-IN (0.6199) |
| **Sarcasm** | 0.6290 +/- 0.0167 | en-AU→en-AU (0.5972) | en-AU→en-IN (0.4742) |

### Combined Training Results

| Task | Test Variety | Combined F1 | Single-Route Avg | Improvement |
|------|--------------|-------------|------------------|-------------|
| Sentiment | en-AU | 0.8407 | 0.6948 | +0.1459 |
| Sentiment | en-IN | 0.7972 | 0.6806 | +0.1166 |
| Sentiment | en-UK | 0.8954 | 0.7644 | +0.1310 |
| Sarcasm | en-AU | 0.6881 | 0.5545 | +0.1337 |
| Sarcasm | en-IN | 0.5265 | 0.5147 | +0.0118 |
| Sarcasm | en-UK | 0.6155 | 0.5612 | +0.0543 |

### Advanced Metrics

| Task | Macro F1 | Accuracy | Perplexity | ECE | Mean Confidence |
|------|----------|----------|------------|-----|-----------------|
| Sentiment | 0.8414 | 0.8415 | 1.4607 | 0.0479 | 0.8800 |
| Sarcasm | 0.6327 | 0.7251 | 1.5957 | 0.1855 | 0.8224 |

## Quick Start

```bash
# Install dependencies
pip install datasets transformers accelerate scikit-learn matplotlib seaborn pandas numpy peft
```

Then open `Q2_2_DeBERTa_Experiments.ipynb` and click **Run All**.

## Configuration (Cell 2)

```python
MODEL_NAME = 'microsoft/deberta-v3-base'
BATCH_SIZE_TRAIN = 16          # Optimized for A4000
BATCH_SIZE_EVAL = 32
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 32
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
LEARNING_RATE_LORA = 2e-4      # Higher LR for LoRA
USE_FP16 = False               # Disabled for stability
SEEDS = [42, 123, 456]
```

## Key Findings

1. **LoRA significantly outperforms Full fine-tuning** for DeBERTa due to training instability with full parameter updates
2. **Combined training beats single-variety** by 11-15% on Sentiment
3. **Sarcasm is harder than Sentiment** (F1 gap of ~0.21)
4. **en-IN is hardest variety** for both tasks
5. **Transfer gap**: Sentiment (1.9%) vs Sarcasm (5.9%)

## Outputs

```
outputs_deberta/
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
| Full (3 seeds, both tasks) | ~6-7 hours |
| Quick (`SEEDS=[42]`) | ~1.5 hours |

See main [README.md](README.md) for detailed documentation.
