"""
utils/zero_shot.py - Zero-shot baseline (frozen base model, no adapter).
Usage:
    from utils.zero_shot import run_zero_shot_baseline
    zs_results = run_zero_shot_baseline(CONFIG, tokenizer, tokenized)
"""

import os, time, torch
import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report)
from transformers import (AutoModelForSequenceClassification, DataCollatorWithPadding)

def run_zero_shot_baseline(config, tokenizer, tokenized):
    """
    Evaluate the frozen base model (no adapter / no fine-tuning)
    on every test set.  This gives a lower-bound baseline.

    Returns
    -------
    zs_results : list[dict]
    """
    print("=" * 62)
    print(f"ZERO-SHOT BASELINE  ({config['base_model']})")
    print("=" * 62)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"], num_labels=2,
        torch_dtype=config["compute_dtype"],
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "single_label_classification"
    model.eval()

    test_varieties = [v for v in config["varieties"] if v in tokenized and "test" in tokenized[v]]
    if "combined" in tokenized and "test" in tokenized["combined"]:
        test_varieties.append("combined")

    # Resume support: check for existing zero-shot results in evaluation cache
    model_key = config["model_key"]
    results_sub = os.path.join(config["results_dir"], config["results_subdir"])
    cache_path = os.path.join(results_sub, f"{model_key}_evaluation_cache.json")
    
    if os.path.exists(cache_path):
        import json
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_res = json.load(f)
        zs_relevant = [r for r in cached_res if r["variety_train"] == "zero-shot"]
        
        # Verify if all requested test varieties are in the cache
        cached_test_vars = set(r["variety_test"] for r in zs_relevant)
        if set(test_varieties).issubset(cached_test_vars):
            print(f"  ✓ Loading {len(zs_relevant)} zero-shot results from cache: {cache_path}")
            return zs_relevant

    zs_results = []

    for test_v in test_varieties:
        print(f"  Zero-shot -> {test_v} test … ", end="", flush=True)

        dl = torch.utils.data.DataLoader(
            tokenized[test_v]["test"], batch_size=32,
            collate_fn=DataCollatorWithPadding(tokenizer),
        )

        all_preds, all_labels = [], []
        t0 = time.perf_counter()
        with torch.no_grad():
            for batch in dl:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                all_preds.extend(torch.argmax(out.logits, dim=-1).cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        elapsed = time.perf_counter() - t0

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        pcf1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
        pcpr = precision_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
        pcre = recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        mf1 = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)

        zs_results.append({
            "variety_train":          "zero-shot",
            "variety_test":           test_v,
            "macro_f1":               mf1,
            "precision_macro":        round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "recall_macro":           round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "accuracy":               round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_sarcasm":             round(float(pcf1[1]), 4),
            "f1_non_sarcasm":         round(float(pcf1[0]), 4),
            "precision_sarcasm":      round(float(pcpr[1]), 4),
            "precision_non_sarcasm":  round(float(pcpr[0]), 4),
            "recall_sarcasm":         round(float(pcre[1]), 4),
            "recall_non_sarcasm":     round(float(pcre[0]), 4),
            "confusion_matrix":       cm.tolist(),
            "classification_report":  classification_report(
                y_true, y_pred,
                target_names=["Non-Sarcastic", "Sarcastic"],
                zero_division=0,
            ),
            "inference_time_s":       round(elapsed, 3),
            "n_test_samples":         len(y_true),
            "y_true":                 y_true.tolist(),
            "y_pred":                 y_pred.tolist(),
        })
        print(f"Macro-F1 = {mf1:.4f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone: {len(zs_results)} zero-shot evaluations")
    return zs_results
