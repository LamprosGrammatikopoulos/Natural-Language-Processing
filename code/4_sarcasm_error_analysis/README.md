## How to Run

```python
%run -i 4_sarcasm_error_analysis/Q4_Sarcasm_Error_Analysis.ipynb
```

# Q4 — Sarcasm Explanation & Error Analysis

Member 2 (Nikolas Ziartis, URN 6911106).

Uses errors from the TinyLlama LoRA adapters (Q2.3) and tests
zero-shot vs few-shot prompting on TinyLlama-1.1B-Chat-v1.0 to
recover them.

## Files
- `Q4_Sarcasm_Error_Analysis.ipynb` — main notebook (run top to bottom)
- `error_analysis.csv` — error data from Q2.3 adapters
- `q4_results.png` — comparison plot used in the report

## Results summary
- Original LoRA adapters: 0/6 (by definition)
- Zero-shot prompting: 3/6
- Few-shot prompting (4-shot CoT): 4/6
