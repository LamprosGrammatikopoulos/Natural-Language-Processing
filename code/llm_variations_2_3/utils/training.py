"""
utils/training.py - Unified training loop for LoRA & full fine-tuning.
Usage:
    from utils.training import train_all_adapters
    adapter_registry, training_log = train_all_adapters(
        CONFIG, tokenizer, tokenized, lora_cfg # lora_cfg may be None
    )
"""

import copy, os, json, time, torch
import numpy as np, pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score)
from transformers import (AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, TrainerCallback)
from peft import get_peft_model

# Helpers used inside the Trainer 
class IntLabelDataCollator(DataCollatorWithPadding):
    """Ensure labels are always Long (int64)."""
    def __call__(self, features):
        batch = super().__call__(features)
        if "labels" in batch:
            batch["labels"] = batch["labels"].to(torch.long)
        return batch

class TestEvaluationCallback(TrainerCallback):
    """Run test-set evaluation at the end of each epoch."""
    def __init__(self, trainer, test_dataset):
        self.trainer = trainer
        self.test_dataset = test_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        self.trainer.evaluate(eval_dataset=self.test_dataset,
                              metric_key_prefix="test")
        return control

def compute_metrics(eval_pred):
    """Compute macro-F1, per-class F1/precision/recall, accuracy."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels = labels.astype(int)
    macro_f1 = f1_score(labels, preds, average="macro",
                        zero_division=0)
    prec_mac = precision_score(labels, preds, average="macro",
                               zero_division=0)
    rec_mac = recall_score(labels, preds, average="macro",
                           zero_division=0)
    acc = accuracy_score(labels, preds)
    pcf1 = f1_score(labels, preds, average=None,
                    zero_division=0, labels=[0, 1])
    pcpr = precision_score(labels, preds, average=None,
                           zero_division=0, labels=[0, 1])
    pcre = recall_score(labels, preds, average=None,
                        zero_division=0, labels=[0, 1])
    return {
        "macro_f1":         float(macro_f1),
        "precision_macro":  float(prec_mac),
        "recall_macro":     float(rec_mac),
        "accuracy":         float(acc),
        "f1_non_sarcasm":   float(pcf1[0]),
        "f1_sarcasm":       float(pcf1[1]),
        "precision_sarcasm": float(pcpr[1]),
        "recall_sarcasm":    float(pcre[1]),
    }

# Single-run training function 
def _train_single_run(variety, run_id, base_model_ref,
                      config, tokenizer, tokenized, lora_cfg):
    """Train one adapter (or full-FT checkpoint) and return
    (adapter_dir, elapsed_seconds, best_val_macro_f1).
    """
    print(f"\n{'='*62}")
    print(f"  Training | Variety: {variety} | Run {run_id}"
          f" | Mode: {config['training_mode']}")
    print(f"{'='*62}")

    model = copy.deepcopy(base_model_ref)
    if torch.cuda.is_available():
        model = model.to(config["compute_dtype"]).cuda()
    model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA if applicable
    if lora_cfg is not None:
        model = get_peft_model(model, lora_cfg)

    adapter_dir = os.path.join(
        config["output_dir"],
        f"{variety.replace('-', '_')}_run{run_id}",
    )

    args = TrainingArguments(
        output_dir=adapter_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"] * 2,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=config["use_fp16"],
        bf16=config["use_bf16"],
        seed=config["seed"] + run_id,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized[variety]["train"],
        eval_dataset=tokenized[variety]["val"],
        data_collator=IntLabelDataCollator(tokenizer),
        compute_metrics=compute_metrics,
    )

    # Optional test-set callback
    if "test" in tokenized[variety]:
        test_cb = TestEvaluationCallback(
            trainer, tokenized[variety]["test"]
        )
        trainer.add_callback(test_cb)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    history = trainer.state.log_history
    with open(os.path.join(adapter_dir, "train_log.json"), "w") as f:
        json.dump(history, f, indent=2)

    best_f1 = max(
        (e.get("eval_macro_f1", 0) for e in history), default=0
    )
    print(f"  Best val Macro-F1 : {best_f1:.4f}")

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return adapter_dir, elapsed, best_f1


# Public API 
def train_all_adapters(config, tokenizer, tokenized, lora_cfg):
    """
    Train ``num_runs`` adapters for each variety in CONFIG.
    Supports resume: existing checkpoints are skipped.

    Returns
    -------
    adapter_registry  : dict[str, (path, best_f1)]
    training_log      : list[dict]
    """
    model_key   = config["model_key"]
    results_sub = os.path.join(config["results_dir"], config["results_subdir"])
    os.makedirs(results_sub, exist_ok=True)

    adapter_registry = {}
    training_log = []

    # Resume support
    log_csv = os.path.join(results_sub, f"{model_key}_training_runs.csv")
    if os.path.exists(log_csv):
        existing = pd.read_csv(log_csv).to_dict("records")
        for row in existing:
            training_log.append({
                "variety":           row["variety"],
                "run":               int(row["run"]),
                "adapter_path":      row["adapter_path"],
                "time_s":            float(row["time_s"]),
                "best_val_macro_f1": float(row["best_val_macro_f1"]),
            })
        print(f"Loaded {len(training_log)} existing run(s) "
              f"from {log_csv}")

    # Determine which file signals a complete checkpoint
    if config["training_mode"] == "lora":
        checkpoint_file = "adapter_model.safetensors"
    else:
        checkpoint_file = "model.safetensors"

    varieties_to_train = [
        v for v in config["varieties"] if v in tokenized
    ]

    for variety in varieties_to_train:
        runs_needed = []
        run_results = []

        for run_id in range(1, config["num_runs"] + 1):
            adapter_dir = os.path.join(
                config["output_dir"],
                f"{variety.replace('-', '_')}_run{run_id}",
            )
            ckpt = os.path.join(adapter_dir, checkpoint_file)
            log_file = os.path.join(adapter_dir, "train_log.json")

            if os.path.exists(ckpt) and os.path.exists(log_file):
                with open(log_file) as fh:
                    saved_log = json.load(fh)
                best_f1 = max(
                    (e.get("eval_macro_f1", e.get("test_macro_f1", 0)) for e in saved_log),
                    default=0
                )
                run_results.append((adapter_dir, 0.0, best_f1))

                existing_entry = next(
                    (e for e in training_log
                    if e["variety"] == variety and int(e["run"]) == run_id),
                    None
                )
                if existing_entry is not None:
                    # Refresh stale f1 value from the actual log file
                    existing_entry["best_val_macro_f1"] = round(best_f1, 4)
                    existing_entry["adapter_path"] = adapter_dir
                else:
                    training_log.append({
                        "variety": variety, "run": run_id,
                        "adapter_path": adapter_dir,
                        "time_s": 0.0,
                        "best_val_macro_f1": round(best_f1, 4),
                    })
                print(f"  ✓ {variety} run {run_id} exists "
                      f"(val F1={best_f1:.4f}) - skipping")
            else:
                runs_needed.append(run_id)

        if runs_needed:
            print(f"\nLoading base model for {variety} "
                  f"(runs: {runs_needed}) …")
            _base = AutoModelForSequenceClassification \
                .from_pretrained(
                    config["base_model"],
                    num_labels=2,
                    torch_dtype=config["compute_dtype"],
                )
            _base.config.pad_token_id = tokenizer.pad_token_id
            _base.config.problem_type = "single_label_classification"
            _base = _base.cpu().eval()
            print("  Base model ready on CPU")

            for run_id in runs_needed:
                path, elapsed, best_f1 = _train_single_run(
                    variety, run_id, _base,
                    config, tokenizer, tokenized, lora_cfg,
                )
                run_results.append((path, elapsed, best_f1))
                training_log.append({
                    "variety": variety, "run": run_id,
                    "adapter_path": path,
                    "time_s": round(elapsed, 1),
                    "best_val_macro_f1": round(best_f1, 4),
                })

            del _base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"  All {config['num_runs']} runs for "
                  f"{variety} already trained")

        best_run = max(run_results, key=lambda x: x[2])
        adapter_registry[variety] = (best_run[0], best_run[2])
        print(f"  [{variety}] Best: {best_run[0]}  "
              f"(F1={best_run[2]:.4f})")

    # Print comparison table 
    print("\n" + "=" * 75)
    print("TRAINING RUN COMPARISON")
    print("=" * 75)
    hdr = (f"{'Variety':<10} {'Run':>4}  {'Val Macro-F1':>13}  "
           f"{'Time (s)':>9}  {'Selected':>10}")
    print(hdr)
    print("-" * 75)

    for entry in training_log:
        v = entry["variety"]
        r = entry["run"]
        f1 = entry["best_val_macro_f1"]
        t = entry["time_s"]
        best = adapter_registry.get(v, (None,))[0]
        sel = "◆ BEST" if best and f"run{r}" in str(best) else ""
        print(f"  {v:<10} {r:>4}  {f1:>13.4f}  {t:>9.1f}  {sel}")
        if r == config["num_runs"]:
            scores = [e["best_val_macro_f1"]
                      for e in training_log if e["variety"] == v]
            print(f"  {'':10} {'range':>4}  "
                  f"{min(scores):.4f} – {max(scores):.4f}  "
                  f"(spread={max(scores)-min(scores):.4f})")
            print()

    print("-" * 75)
    print("◆ = adapter selected for cross-variety evaluation")

    # Save CSV
    # Load existing selected values before overwriting
    existing_selected = {}
    if os.path.exists(log_csv):
        try:
            _prev = pd.read_csv(log_csv)
            for _, r in _prev.iterrows():
                existing_selected[(r["variety"], int(r["run"]))] = bool(r.get("selected", False))
        except Exception:
            pass

    df = pd.DataFrame(training_log)
    df["selected"] = df.apply(
        lambda row: (
            f"run{int(row['run'])}" in str(adapter_registry.get(row["variety"], (None,))[0])
            if row["variety"] in adapter_registry
            else existing_selected.get((row["variety"], int(row["run"])), False)
        ),
        axis=1,
    )
    df.to_csv(log_csv, index=False)
    print(f"\nTraining runs saved -> {log_csv}")

    # Summary
    print("\n" + "=" * 62)
    print("ALL ADAPTERS READY")
    print("=" * 62)
    for v, (path, f1) in adapter_registry.items():
        print(f"  {v:<10}  Best: {path}  (val F1={f1:.4f})")

    return adapter_registry, training_log