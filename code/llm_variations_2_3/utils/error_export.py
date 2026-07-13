"""
utils/error_export.py - Per-sample error export with text + heatmaps.

Usage:
    from utils.error_export import export_error_analysis
    export_error_analysis(CONFIG, eval_results, tokenized,
                          adapter_registry, tokenizer, variety_data)
"""
import os, time, torch
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (AutoModelForSequenceClassification, DataCollatorWithPadding)
from peft import PeftModel

def export_error_analysis(config, eval_results, combined_eval_results, tokenized, adapter_registry, tokenizer, variety_data):
    """
    For each adapter x test-set pair, collect per-sample
    predictions and export mispredictions to CSV.
    Also generate a 4x4 error-rate heatmap.
    """
    model_key   = config["model_key"]
    results_sub = os.path.join(config["results_dir"], config["results_subdir"])
    os.makedirs(results_sub, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_errors = []

    test_varieties = [v for v in config["varieties"] if v in tokenized and "test" in tokenized[v]]
    if "combined" in tokenized and "test" in tokenized["combined"]:
        test_varieties.append("combined")

    # Process directly from cached evaluation results instead of running inference again!
    all_evals = eval_results + (combined_eval_results if combined_eval_results else [])
    
    for res in all_evals:
        train_v = res["variety_train"]
        test_v = res["variety_test"]
        
        preds_list = res["y_pred"]
        labels_list = res["y_true"]
        
        # Get raw texts from variety_data
        text_col = config.get("text_col", "text")
        if test_v in variety_data and "test" in variety_data[test_v]:
            texts = variety_data[test_v]["test"][text_col].tolist()
        else:
            texts = ["[text unavailable]"] * len(labels_list)

        label_map = {0: "Non-Sarcastic", 1: "Sarcastic"}

        for i, (pred, true) in enumerate(zip(preds_list, labels_list)):
            if pred != true:
                all_errors.append({
                    "model":             model_key,
                    "adapter_variety":   train_v,
                    "test_variety":      test_v,
                    "sample_idx":        i,
                    "text":              texts[i] if i < len(texts) else "",
                    "true_label":        label_map[int(true)],
                    "predicted_label":   label_map[int(pred)],
                })

    # Save error CSV
    error_path = os.path.join(results_sub, f"{model_key}_error_analysis.csv")
    pd.DataFrame(all_errors).to_csv(error_path, index=False)
    print(f"Error analysis saved -> {error_path}   ({len(all_errors)} errors)")

    # Error-rate heatmap 
    _plot_error_heatmap(config, all_evals, results_sub, model_key)

    return all_errors

def _plot_error_heatmap(config, eval_results, results_sub, model_key):
    """Generate a 4×4 error-rate heatmap from eval_results."""
    varieties = config["varieties"]
    all_labels = varieties + (["combined"]
                              if any(r["variety_test"] == "combined"
                                     or r["variety_train"] == "combined"
                                     for r in eval_results)
                              else [])
    n = len(all_labels)
    err_matrix = np.full((n, n), np.nan)

    for r in eval_results:
        tv = r["variety_train"]
        te = r["variety_test"]
        if tv in all_labels and te in all_labels:
            i = all_labels.index(tv)
            j = all_labels.index(te)
            err_matrix[i, j] = round(1.0 - r["accuracy"], 4)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(err_matrix, annot=True, fmt=".3f",
                cmap="YlOrRd",
                xticklabels=all_labels,
                yticklabels=all_labels,
                ax=ax, vmin=0, vmax=0.5)
    ax.set_xlabel("Test Variety")
    ax.set_ylabel("Adapter Trained On")
    ax.set_title(f"{model_key} - Error Rate Heatmap", fontweight="bold")
    plt.tight_layout()

    hm_path = os.path.join(results_sub, f"{model_key}_error_heatmap.png")
    plt.savefig(hm_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Error heatmap saved -> {hm_path}")
