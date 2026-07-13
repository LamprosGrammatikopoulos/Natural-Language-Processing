"""
utils/evaluation.py - Cross-variety evaluation matrix.

Usage:
    from utils.evaluation import evaluate_cross_variety
    eval_results = evaluate_cross_variety(
        CONFIG, tokenizer, tokenized, adapter_registry
    )
"""
import os, time, torch
import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report)
from transformers import (AutoModelForSequenceClassification, DataCollatorWithPadding)
from peft import PeftModel

def _evaluate_single(adapter_dir, test_ds, variety_train, variety_test, config, tokenizer):
    """
    Load the model (base+adapter or full-FT checkpoint),
    run inference on *test_ds*, return a results dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["training_mode"] == "lora":
        # LoRA: load base then overlay adapter
        model = AutoModelForSequenceClassification.from_pretrained(
            config["base_model"], num_labels=2,
            torch_dtype=config["compute_dtype"],
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(model, adapter_dir)
    else:
        # Full fine-tuning: load from checkpoint dir
        model = AutoModelForSequenceClassification.from_pretrained(
            adapter_dir, num_labels=2,
            torch_dtype=config["compute_dtype"],
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    dataloader = torch.utils.data.DataLoader(test_ds, batch_size=32, collate_fn=DataCollatorWithPadding(tokenizer))

    all_preds, all_labels = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            all_preds.extend(torch.argmax(out.logits, dim=-1).cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    elapsed = time.perf_counter() - t0

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    pcf1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
    pcpr = precision_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
    pcre = recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "variety_train":          variety_train,
        "variety_test":           variety_test,
        "precision_sarcasm":      round(float(pcpr[1]), 4),
        "precision_non_sarcasm":  round(float(pcpr[0]), 4),
        "recall_sarcasm":         round(float(pcre[1]), 4),
        "recall_non_sarcasm":     round(float(pcre[0]), 4),
        "f1_non_sarcasm":         round(float(pcf1[0]), 4),
        "f1_sarcasm":             round(float(pcf1[1]), 4),
        "accuracy":               round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1":               round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "precision_macro":        round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro":           round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix":       cm.tolist(),
        "classification_report":  classification_report(y_true, y_pred, target_names=["Non-Sarcastic", "Sarcastic"], zero_division=0),
        "inference_time_s":       round(elapsed, 3),
        "n_test_samples":         len(y_true),
        "y_true":                 y_true.tolist(),
        "y_pred":                 y_pred.tolist(),
    }

#  Public API 
def evaluate_cross_variety(config, tokenizer, tokenized, adapter_registry):
    """
    Run the full cross-variety evaluation matrix:
    each adapter is tested against every test split.

    Returns
    -------
    eval_results : list[dict]
    """
    eval_results = []

    # Resume support: check for existing evaluation cache
    model_key = config["model_key"]
    results_sub = os.path.join(config["results_dir"], config["results_subdir"])
    cache_path = os.path.join(results_sub, f"{model_key}_evaluation_cache.json")
    
    if os.path.exists(cache_path):
        import json
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_res = json.load(f)
        
        # Filter for the relevant varieties being evaluated in this session
        requested_train_vars = set(map(str, config["varieties"]))
        relevant_res = [r for r in cached_res if str(r["variety_train"]) in requested_train_vars]
        
        # Verify the cache is complete for the requested training varieties
        # Each train variety should have evaluated against all test varieties in the cache
        actual_train_vars = set(r["variety_train"] for r in relevant_res)
        if requested_train_vars.issubset(actual_train_vars):
            print(f"  ✓ Loading {len(relevant_res)} evaluation results from cache: {cache_path}")
            return relevant_res
        else:
            print(f"  ! Cache incomplete (found {len(actual_train_vars)}/{len(requested_train_vars)} train varieties). Running inference.")

    # Build the list of test varieties: everything available in tokenized
    test_varieties = []
    seen = set()
    # Priority order: Config-specified varieties, then 'combined', then any other variety in tokenized
    for v in config.get("varieties", []) + ["combined"] + list(tokenized.keys()):
        if v in tokenized and "test" in tokenized[v] and v not in seen:
            test_varieties.append(v)
            seen.add(v)

    for train_v in adapter_registry.keys():
        adapter_dir, _ = adapter_registry[train_v]

        for test_v in test_varieties:
            if test_v not in tokenized:
                continue
            if "test" not in tokenized[test_v]:
                continue
            tag = "(in-dist)" if train_v == test_v else "(cross)  "
            print(f"  {train_v} adapter -> {test_v} test {tag} … ", end="", flush=True)
            res = _evaluate_single(adapter_dir, tokenized[test_v]["test"], train_v, test_v, config, tokenizer)
            eval_results.append(res)
            print(f"Macro-F1 = {res['macro_f1']:.4f}")

    print(f"\nDone: {len(eval_results)} evaluations")
    return eval_results
