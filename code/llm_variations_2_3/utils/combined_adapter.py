"""
utils/combined_adapter.py — Train & evaluate the combined (pooled) adapter.

Usage:
    from utils.combined_adapter import run_combined_experiment
    combined_eval_results = run_combined_experiment(CONFIG, tokenizer, tokenized, lora_cfg, adapter_registry)
"""
from utils.training import train_all_adapters
from utils.evaluation import evaluate_cross_variety

def run_combined_experiment(config, tokenizer, tokenized, lora_cfg, adapter_registry):
    """
    Train combined adapter using the 'combined' tokenized splits,
    then evaluate it against all test sets.

    *adapter_registry* is updated in-place with the 'combined' entry.

    Returns
    -------
    combined_eval_results : list[dict]
    """
    if "combined" not in tokenized:
        print("No 'combined' data available — skipping.")
        return []

    # Re-use the standard training pipeline with varieties=['combined']
    combined_config = {**config, "varieties": ["combined"]}

    comb_registry, _ = train_all_adapters(combined_config, tokenizer, tokenized, lora_cfg,)

    # Merge into the main adapter_registry
    adapter_registry.update(comb_registry)

    # Evaluate combined adapter against all test sets
    print("\nEvaluating combined adapter against all test sets …")
    combined_eval_results = evaluate_cross_variety(combined_config, tokenizer, tokenized, comb_registry,)

    return combined_eval_results
