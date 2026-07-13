"""
utils/results.py - Results tables, confusion matrices, CSV export.

Usage:
    from utils.results import generate_all_results
    generate_all_results(CONFIG, eval_results,
                         combined_eval_results,
                         adapter_registry, training_log,
                         total_params, zs_results=None)
"""
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_all_results(config, eval_results, combined_eval_results=None, adapter_registry=None, training_log=None, total_params=None, zs_results=None):
    """
    Print full results table, classification reports,
    confusion matrix plots, transferability analysis,
    and save all CSVs + PNGs.
    """
    model_key = config["model_key"]
    results_sub = os.path.join(config["results_dir"], config["results_subdir"])
    os.makedirs(results_sub, exist_ok=True)

    # Merge per-variety + combined eval results
    # 1. Start clean
    all_res = list(eval_results)

    # 2. Merge combined results
    if combined_eval_results:
        all_res += combined_eval_results

    # 3. Deduplicate (GLOBAL fix)
    unique_res = {}
    for r in all_res:
        key = (r["variety_train"], r["variety_test"])
        unique_res[key] = r  # keeps LAST occurrence

    all_res = list(unique_res.values())

    # 1. Full results table 
    print("\n" + "=" * 90)
    print("CROSS-VARIETY EVALUATION - FULL RESULTS TABLE")
    print("=" * 90)
    header = (f"{'Train Adapter':<14} {'Test Set':<10} "
              f"{'Macro-F1':>9} {'Prec(mac)':>10} "
              f"{'Rec(mac)':>9} {'F1-Sarc':>8} "
              f"{'F1-NonSarc':>11} {'Acc':>7}")
    print(header)
    print("-" * 90)

    for r in sorted(all_res, key=lambda x: (x["variety_train"], x["variety_test"])):
        in_d = " ◆" if r["variety_train"] == r["variety_test"] else "  "
        f1s = r.get("f1_sarcasm", r.get("f1_sarcastic", 0.0))
        f1n = r.get("f1_non_sarcasm", r.get("f1_non_sarcastic", 0.0))
        print(f"{r['variety_train']:<14} {r['variety_test']:<10} "
              f"{r['macro_f1']:>9.4f} {r['precision_macro']:>10.4f} "
              f"{r['recall_macro']:>9.4f} {f1s:>8.4f} "
              f"{f1n:>11.4f} {r['accuracy']:>7.4f}{in_d}")

    print("-" * 90)
    print("◆ = In-distribution (same variety for train and test)")

    # 2. Per-class classification reports (in-dist) 
    print("\nDETAILED CLASSIFICATION REPORTS (In-Distribution Only)")
    for r in all_res:
        if r["variety_train"] != r["variety_test"]:
            continue
        print(f"\n{'=' * 55}")
        print(f"  Adapter: {r['variety_train']}  |   Test: {r['variety_test']}")
        print(f"{'=' * 55}")
        print(r["classification_report"])
        f1s = r.get("f1_sarcasm", r.get("f1_sarcastic", 0.0))
        ps = r.get("precision_sarcasm", r.get("precision_sarcastic", 0.0))
        rs = r.get("recall_sarcasm", r.get("recall_sarcastic", 0.0))
        print(f"  ★ Macro-F1       : {r['macro_f1']:.4f}")
        print(f"     Sarcasm F1     : {f1s:.4f}   (key metric - avoids majority-class trap)")
        print(f"     Sarcasm Prec.  : {ps:.4f}")
        print(f"     Sarcasm Recall : {rs:.4f}")

    # 3. Confusion matrices (in-dist, including combined->combined)
    in_dist = [r for r in all_res if r["variety_train"] == r["variety_test"]]

    # Remove duplicates: keep only one entry per train/test pair
    seen_pairs = set()
    unique_in_dist = []
    for r in in_dist:
        pair = (r["variety_train"], r["variety_test"])
        if pair not in seen_pairs:
            unique_in_dist.append(r)
            seen_pairs.add(pair)
    in_dist = unique_in_dist

    # Sort so that combined adapter appears last
    in_dist.sort(key=lambda r: (r["variety_train"] != "combined", r["variety_train"]))
    
    if in_dist:
        n = len(in_dist)
        fig, axes = plt.subplots(1, n,
                                 figsize=(5 * n, 4.5))
        if n == 1:
            axes = [axes]
        fig.suptitle(
            "Confusion Matrices - In-Distribution (Normalised)",
            fontsize=13, fontweight="bold",
        )
        for ax, res in zip(axes, in_dist):
            cm = np.array(res["confusion_matrix"])
            cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Non-Sarc", "Sarcastic"])
            ax.set_yticklabels(["Non-Sarc", "Sarcastic"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            f1s = res.get("f1_sarcasm", res.get("f1_sarcastic", 0.0))
            ax.set_title(
                f"{res['variety_train']} -> "
                f"{res['variety_test']}\n"
                f"Macro-F1={res['macro_f1']:.4f} | "
                f"F1-Sarc={f1s:.4f}",
                fontsize=10, fontweight="bold",
            )
            plt.colorbar(im, ax=ax, fraction=0.046)
            for i in range(2):
                for j in range(2):
                    col = "white" if cmn[i, j] > 0.5 else "black"
                    ax.text(j, i, f"{cm[i,j]}\n({cmn[i,j]:.1%})", ha="center", va="center", color=col, fontsize=11, fontweight="bold")
        plt.tight_layout()
        cm_path = os.path.join(results_sub, f"{model_key}_confusion_matrices.png",)
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Confusion matrices saved -> {cm_path}")

    # 4. Transferability analysis 
    print("\nTRANSFERABILITY ANALYSIS")
    print("=" * 65)

    varieties_present = [v for v in config["varieties"] if v in (adapter_registry or {})]
    train_vars = varieties_present[:]
    if combined_eval_results:
        train_vars.append("combined")

    for tv in train_vars:
        in_f1 = next(
            (r["macro_f1"] for r in all_res
             if r["variety_train"] == tv
             and r["variety_test"] == tv), None,
        )
        if in_f1 is None:
            continue
        cross = [r["macro_f1"] for r in all_res
                 if r["variety_train"] == tv
                 and r["variety_test"] != tv]
        avg_c = np.mean(cross) if cross else 0.0
        drop = in_f1 - avg_c
        pct = 100 * drop / in_f1 if in_f1 > 0 else 0
        print(f"\n  {tv} Adapter:")
        print(f"    In-distribution Macro-F1  : {in_f1:.4f}")
        print(f"    Avg cross-variety F1       : {avg_c:.4f}")
        print(f"    Performance drop           : {drop:.4f}  ({pct:.1f}%)")

    # Inner-circle ↔ Inner-circle
    uk_au = next((r["macro_f1"] for r in all_res
                  if r["variety_train"] == "en-UK"
                  and r["variety_test"] == "en-AU"), None)
    au_uk = next((r["macro_f1"] for r in all_res
                  if r["variety_train"] == "en-AU"
                  and r["variety_test"] == "en-UK"), None)
    if uk_au is not None and au_uk is not None:
        print("\n  Inner-circle ↔ Inner-circle transfer:")
        print(f"    en-UK -> en-AU : {uk_au:.4f}")
        print(f"    en-AU -> en-UK : {au_uk:.4f}")
        print(f"    Average        : {(uk_au+au_uk)/2:.4f}")

    # Inner -> Outer
    uk_in = next((r["macro_f1"] for r in all_res
                  if r["variety_train"] == "en-UK"
                  and r["variety_test"] == "en-IN"), None)
    au_in = next((r["macro_f1"] for r in all_res
                  if r["variety_train"] == "en-AU"
                  and r["variety_test"] == "en-IN"), None)
    if uk_in is not None and au_in is not None:
        print("\n  Inner-circle -> Outer-circle transfer:")
        print(f"    en-UK -> en-IN : {uk_in:.4f}")
        print(f"    en-AU -> en-IN : {au_in:.4f}")
        print(f"    Average        : {(uk_in+au_in)/2:.4f}")

    # 5. Save CSVs 
    # 5a. In-distribution metrics
    indist_rows = []
    for r in eval_results:
        if r["variety_train"] == r["variety_test"]:
            indist_rows.append({
                "variety":               r["variety_train"],
                "macro_f1":              r["macro_f1"],
                "precision_macro":       r["precision_macro"],
                "recall_macro":          r["recall_macro"],
                "accuracy":              r["accuracy"],
                "f1_sarcasm":            r.get("f1_sarcasm", r.get("f1_sarcastic", 0.0)),
                "f1_non_sarcasm":        r["f1_non_sarcasm"],
                "precision_sarcasm":     r.get("precision_sarcasm", r.get("precision_sarcastic", 0.0)),
                "precision_non_sarcasm": r["precision_non_sarcasm"],
                "recall_sarcasm":        r.get("recall_sarcasm", r.get("recall_sarcastic", 0.0)),
                "recall_non_sarcasm":    r["recall_non_sarcasm"],
                "n_test_samples":        r["n_test_samples"],
                "inference_time_s":      r["inference_time_s"],
            })
    p1 = os.path.join(results_sub, f"{model_key}_indist_metrics.csv")
    pd.DataFrame(indist_rows).to_csv(p1, index=False)
    print(f"\nIn-distribution metrics -> {p1}")

    # 5b. Transferability
    xfer_rows = []
    for tv in varieties_present:
        in_f1 = next((r["macro_f1"] for r in eval_results if r["variety_train"] == tv and r["variety_test"] == tv), None)
        for r in eval_results:
            if r["variety_train"] != tv:
                continue
            xfer_rows.append({
                "adapter_trained_on": tv,
                "tested_on":          r["variety_test"],
                "macro_f1":           r["macro_f1"],
                "f1_sarcasm":         r.get("f1_sarcasm", r.get("f1_sarcastic", 0.0)),
                "in_distribution":    r["variety_train"] == r["variety_test"],
                "f1_drop_vs_indist":  round(in_f1 - r["macro_f1"], 4) if in_f1 else None,
            })
    p2 = os.path.join(results_sub, f"{model_key}_transferability.csv")
    pd.DataFrame(xfer_rows).to_csv(p2, index=False)
    print(f"Transferability matrix -> {p2}")    # 5c. Efficiency summary
    if total_params is not None:
        eff_rows = []
        for r in eval_results:
            if r["variety_train"] == r["variety_test"]:
                adapter_dir = adapter_registry.get(r["variety_train"], (None,))[0] if adapter_registry else None
                adapter_size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, files in os.walk(adapter_dir) for fn in files) / 1e6 if adapter_dir and os.path.exists(adapter_dir) else 0
                if config["training_mode"] == "lora":
                    # Count actual LoRA params from the saved adapter weights
                    _lora_params = 0
                    try:
                        from safetensors import safe_open
                        for _fname in os.listdir(adapter_dir):
                            if _fname.endswith(".safetensors"):
                                with safe_open(os.path.join(adapter_dir, _fname), framework="pt", device="cpu") as _f:
                                    for _key in _f.keys():
                                        _lora_params += _f.get_tensor(_key).numel()
                    except Exception:
                        _lora_params = 0

                    eff_rows.append({
                        "variety":              r["variety_train"],
                        "n_test_samples":       r["n_test_samples"],
                        "total_inference_s":    r["inference_time_s"],
                        "ms_per_sample":        round(r["inference_time_s"] / r["n_test_samples"] * 1000, 2),
                        "base_model_params":    total_params,
                        "lora_trainable_params": _lora_params if _lora_params > 0 else "Unknown",
                        "lora_pct": round(100 * _lora_params / total_params, 4) if _lora_params > 0 else "Unknown",
                        "adapter_size_mb":      f"~{int(adapter_size)}",
                        "base_model_size_gb":   round(total_params * 2 / 1e9, 2),
                    })
                else: 
                    eff_rows.append({
                        "variety":              r["variety_train"],
                        "n_test_samples":       r["n_test_samples"],
                        "total_inference_s":    r["inference_time_s"],
                        "ms_per_sample":        round(r["inference_time_s"] / r["n_test_samples"] * 1000, 2),
                        "base_model_params":    total_params,
                        "lora_trainable_params": "N/A",
                        "lora_pct":             "N/A",
                        "adapter_size_mb":      "N/A",
                        "base_model_size_gb":   round(total_params * 2 / 1e9, 2),
                        "model":                config["base_model"],
                        "fine_tuning":          "full",
                    })
        p3 = os.path.join(results_sub, f"{model_key}_efficiency_summary.csv")
        pd.DataFrame(eff_rows).to_csv(p3, index=False)
        print(f"Efficiency summary     -> {p3}")

    # 5d. Cross-variety results CSV (full 4x4 matrix) 
    _save_cross_variety_csv(all_res, results_sub, model_key)

    # 5e. All predictions CSV 
    _save_all_predictions_csv(all_res, results_sub, model_key)

    # 6. Heatmaps 
    # 6a. Transferability heatmap (Macro-F1)
    _plot_transferability_heatmap(config, all_res, results_sub, model_key)

    # 6b. Cross-variety heatmaps (multi-panel with bar chart)  NEW
    _plot_cross_variety_heatmaps(config, all_res, results_sub, model_key)

    # 6c. Error rate heatmaps (FP/FN/Total) 
    _plot_error_rate_heatmaps(config, all_res, results_sub, model_key)

    # 7. Training curves + epoch metrics 
    _plot_training_curves(config, results_sub, model_key, adapter_registry)

    # 8. Zero-shot results + ladder 
    if zs_results:
        _save_zero_shot_results(zs_results, all_res, config, results_sub, model_key)

    # 9. Save Full Evaluation Cache (for Resume/Modular use)
    cache_path = os.path.join(results_sub, f"{model_key}_evaluation_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    print(f"Evaluation cache saved -> {cache_path}")

    print(f"\nAll CSVs saved to {results_sub}/")

    # 10. Efficiency summary (printed) 
    if total_params is not None:
        print(f"\nLoRA EFFICIENCY SUMMARY")
        print("=" * 60)
        print(f"  Base model params   : {total_params:,}")
        mode_label = ("LoRA adapters" if config["training_mode"]
                      == "lora" else "Full fine-tuning")
        print(f"  Training mode       : {mode_label}")
        print(f"  Base model size     : "
              f"~{total_params * 2 / 1e9:.1f} GB")
        print()
        print("  Inference time per adapter:")
        for r in all_res:
            if r["variety_train"] == r["variety_test"]:
                ms = (r["inference_time_s"] / r["n_test_samples"]) * 1000
                print(f"    {r['variety_train']:<10}: "
                      f"~{ms:.1f} ms / sample  "
                      f"({r['n_test_samples']} test samples "
                      f"in {r['inference_time_s']:.1f}s)")

    # 11. File inventory 
    print("\n" + "=" * 55)
    print(" COMPLETE - FILE INVENTORY")
    print("=" * 55)
    print("\nResults:")
    for f in sorted(os.listdir(results_sub)):
        p = os.path.join(results_sub, f)
        print(f"  {f:<45}  ({os.path.getsize(p)/1024:.1f} KB)" if os.path.isfile(p) else f"  {f}")

    if adapter_registry:
        print("\nAdapters / Checkpoints:")
        for v, (path, f1) in adapter_registry.items():
            size_mb = sum(
                os.path.getsize(os.path.join(dp, fn))
                for dp, _, files in os.walk(path)
                for fn in files
            ) / 1e6
            print(f"  {v:<10}  {path:<50}  ({size_mb:.1f} MB)  val_F1={f1:.4f}")

# Helper functions (restored from v11 functionality)
def _save_cross_variety_csv(all_res, results_sub, model_key):
    """Save the full cross-variety results CSV (all adapter x test combos)."""
    rows = []
    for r in all_res:
        rows.append({k: v for k, v in r.items() if k not in ('confusion_matrix', 'classification_report', 'y_true', 'y_pred')})
    path = os.path.join(results_sub, f"{model_key}_cross_variety_results.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Cross-variety results  -> {path}")

def _save_all_predictions_csv(all_res, results_sub, model_key):
    """Save per-sample predictions for every adapter x test combo."""
    rows = []
    for r in all_res:
        y_true = r.get("y_true", [])
        y_pred = r.get("y_pred", [])
        for idx, (yt, yp) in enumerate(zip(y_true, y_pred)):
            rows.append({
                "adapter_variety": r["variety_train"],
                "test_variety":    r["variety_test"],
                "sample_idx":      idx,
                "true_label":      int(yt),
                "predicted_label": int(yp),
                "correct":         int(yt) == int(yp),
            })
    if rows:
        path = os.path.join(results_sub, f"{model_key}_all_predictions.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"All predictions        -> {path}  ({len(rows)} rows)")
    else:
        print("  (No per-sample predictions available for all_predictions.csv)")

def _plot_transferability_heatmap(config, all_res, results_sub, model_key):
    """Generate a 4x4 Macro-F1 heatmap across all adapter/test pairings."""
    varieties = config["varieties"]
    all_labels = varieties + (["combined"] if any(r["variety_train"] == "combined" for r in all_res) else [])
    
    n = len(all_labels)
    f1_matrix = np.full((n, n), np.nan)

    for r in all_res:
        tv = r["variety_train"]
        te = r["variety_test"]
        if tv in all_labels and te in all_labels:
            i = all_labels.index(tv)
            j = all_labels.index(te)
            f1_matrix[i, j] = r["macro_f1"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(f1_matrix, annot=True, fmt=".3f", cmap="Blues", xticklabels=all_labels, yticklabels=all_labels, ax=ax, vmin=0, vmax=1.0)
    ax.set_xlabel("Test Variety")
    ax.set_ylabel("Adapter Trained On")
    ax.set_title(f"{model_key} - Cross-Variety Macro-F1 Heatmap", fontweight="bold")
    plt.tight_layout()

    hm_path = os.path.join(results_sub, f"{model_key}_transferability_heatmap.png")
    plt.savefig(hm_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Transferability heatmap saved -> {hm_path}")

def _plot_cross_variety_heatmaps(config, all_res, results_sub, model_key):
    """Generate multi-panel cross-variety heatmaps (F1, F1-Sarcasm) + bar chart."""
    varieties = config["varieties"]
    all_labels = varieties + (["combined"] if any(r["variety_train"] == "combined" for r in all_res) else [])
    n_rows = len(all_labels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Cross-Variety Evaluation - {model_key}", fontsize=13, fontweight="bold")

    def build_matrix(metric):
        M = np.full((n_rows, n_rows), np.nan)
        for r in all_res:
            tv, te = r["variety_train"], r["variety_test"]
            if tv in all_labels and te in all_labels:
                i, j = all_labels.index(tv), all_labels.index(te)
                M[i, j] = r.get(metric, r.get(metric.replace("f1_sarcasm", "f1_sarcastic"), np.nan))
        return M

    short_labels = [v.replace("en-", "") for v in all_labels]

    for ax, (metric, title) in zip(axes[:2], [
        ("macro_f1",     "Macro-F1"),
        ("f1_sarcasm", "F1 - Sarcasm Class"),
    ]):
        M = build_matrix(metric)
        im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(n_rows)); ax.set_yticks(range(n_rows))
        ax.set_xticklabels(short_labels, fontsize=12); ax.set_yticklabels(short_labels, fontsize=12)
        ax.set_xlabel("Test Set", fontsize=11); ax.set_ylabel("Adapter (trained on)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(n_rows):
            for j in range(n_rows):
                if not np.isnan(M[i,j]):
                    col = "white" if M[i,j] > 0.6 else "black"
                    ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=13, fontweight="bold", color=col)
                if all_labels[i] == all_labels[j]:
                    import matplotlib.patches as patches
                    ax.add_patch(patches.Rectangle((j-.5, i-.5), 1, 1, fill=False, edgecolor="deepskyblue", lw=3))

    M_f1 = build_matrix("macro_f1")
    in_d  = [M_f1[i, i] if not np.isnan(M_f1[i, i]) else 0 for i in range(n_rows)]
    cross = [max([M_f1[i, j] for j in range(n_rows) if i != j and not np.isnan(M_f1[i, j])] or [0]) for i in range(n_rows)]

    x = np.arange(n_rows)
    w = 0.35
    ax3 = axes[2]
    b1 = ax3.bar(x - w / 2, in_d, w, label="In-distribution", color="#5B9BD5", edgecolor="black")
    b2 = ax3.bar(x + w / 2, cross, w, label="Best cross-variety", color="#ED7D31", edgecolor="black")
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_labels, fontsize=12)
    ax3.set_ylabel("Macro-F1")
    ax3.set_ylim(0, 1.15)
    ax3.set_title("In-distribution vs. Best Cross-Variety", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.yaxis.grid(True, alpha=0.3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(results_sub, f"{model_key}_cross_variety_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Cross-variety heatmaps -> {path}")

def _plot_error_rate_heatmaps(config, all_res, results_sub, model_key):
    """Generate 3-panel error rate heatmaps (FP rate, FN rate, Total error)."""
    varieties = config["varieties"]
    all_labels = varieties + (["combined"] if any(
        r["variety_train"] == "combined" or r["variety_test"] == "combined"
        for r in all_res) else [])
    n = len(all_labels)

    fp_mat = np.full((n, n), np.nan)
    fn_mat = np.full((n, n), np.nan)
    err_mat = np.full((n, n), np.nan)

    for r in all_res:
        tv, te = r["variety_train"], r["variety_test"]
        if tv in all_labels and te in all_labels:
            i, j = all_labels.index(tv), all_labels.index(te)
            y_true = np.array(r.get("y_true", []))
            y_pred = np.array(r.get("y_pred", []))
            total = len(y_true)
            if total == 0:
                continue
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            fp_mat[i, j] = round(fp / total, 4)
            fn_mat[i, j] = round(fn / total, 4)
            err_mat[i, j] = round((fp + fn) / total, 4)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"{model_key} - Error Rate Heatmaps", fontsize=14, fontweight="bold")

    for ax, mat, title, cmap in [
        (axes[0], fp_mat, "False Positive Rate", "YlOrRd"),
        (axes[1], fn_mat, "False Negative Rate", "YlOrRd"),
        (axes[2], err_mat, "Total Error Rate", "YlOrRd"),
    ]:
        sns.heatmap(mat, annot=True, fmt=".3f", cmap=cmap,
                    xticklabels=all_labels, yticklabels=all_labels,
                    ax=ax, vmin=0, vmax=0.5)
        ax.set_xlabel("Test Variety")
        ax.set_ylabel("Adapter Trained On")
        ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(results_sub, f"{model_key}_error_rate_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Error rate heatmaps    -> {path}")

def _plot_training_curves(config, results_sub, model_key, adapter_registry):
    """Plot training curves and save per-epoch metrics CSV."""
    output_dir = config.get("output_dir", "")
    varieties = config["varieties"]
    run_colours = {1: '#2196F3', 2: '#4CAF50', 3: '#FF5722'}
    run_styles = {1: 'o-', 2: 's--', 3: '^:'}
    n_varieties = len(varieties)

    has_data = False
    for variety in varieties:
        for run_id in range(1, config.get("num_runs", 3) + 1):
            log_path = os.path.join(output_dir, f"{variety.replace('-', '_')}_run{run_id}", "train_log.json")
            if os.path.exists(log_path):
                has_data = True
                break
        if has_data:
            break

    if not has_data:
        print("  (No training logs found for training curves)")
        return

    fig, axes = plt.subplots(2, n_varieties, figsize=(16, 9), sharey='row')
    if n_varieties == 1:
        axes = np.array([axes]).reshape(2, 1)

    fig.suptitle(f'{model_key} Validation Metrics during Training - All Runs per Variety',
                 fontsize=13, fontweight='bold')

    for col, variety in enumerate(varieties):
        ax_f1 = axes[0][col]
        ax_loss = axes[1][col]
        ax_f1.set_title(variety, fontweight='bold', fontsize=12)
        for run_id in range(1, config.get("num_runs", 3) + 1):
            log_path = os.path.join(output_dir, f"{variety.replace('-', '_')}_run{run_id}", "train_log.json")
            if not os.path.exists(log_path):
                continue
            with open(log_path) as f:
                log = json.load(f)
            eval_entries = [e for e in log if 'eval_macro_f1' in e or 'test_macro_f1' in e]
            if not eval_entries:
                continue
            epochs = [e['epoch'] for e in eval_entries]
            f1s = [e.get('eval_macro_f1', e.get('test_macro_f1')) for e in eval_entries]
            losses = [e.get('eval_loss', e.get('test_loss')) for e in eval_entries]
            best_path = (adapter_registry or {}).get(variety, (None,))[0]
            lw = 2.5 if best_path and f'run{run_id}' in str(best_path) else 1.2
            label = f'Run {run_id}' + (' ◆' if lw > 2 else '')
            ax_f1.plot(epochs, f1s, run_styles[run_id], label=label, color=run_colours.get(run_id, '#000000'), linewidth=lw)
            if any(l is not None for l in losses):
                ax_loss.plot(epochs, losses, run_styles[run_id], label=label, color=run_colours.get(run_id, '#000000'), linewidth=lw)
        for ax, ylabel in [(ax_f1, 'Macro-F1'), (ax_loss, 'Val Loss')]:
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        ax_f1.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(results_sub, f"{model_key}_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training curves -> {path}")

    # Save per-epoch metrics CSV
    epoch_rows = []
    for variety in varieties:
        for run_id in range(1, config.get("num_runs", 3) + 1):
            log_path = os.path.join(output_dir, f"{variety.replace('-', '_')}_run{run_id}", "train_log.json")
            if not os.path.exists(log_path):
                continue
            with open(log_path) as f:
                log = json.load(f)
            for entry in log:
                if 'eval_macro_f1' in entry or 'test_macro_f1' in entry:
                    epoch_rows.append({
                        'variety': variety, 'run': run_id,
                        'epoch': entry['epoch'],
                        'eval_macro_f1': round(entry.get('eval_macro_f1', entry.get('test_macro_f1', float('nan'))), 4),
                        'eval_loss': round(entry.get('eval_loss', entry.get('test_loss', float('nan'))), 4),
                        'eval_f1_sarcasm': round(entry.get('eval_f1_sarcasm', entry.get('eval_f1_sarcastic', entry.get('test_f1_sarcasm', entry.get('test_f1_sarcastic', float('nan'))))), 4),
                        'eval_accuracy': round(entry.get('eval_accuracy', entry.get('test_accuracy', float('nan'))), 4),
                    })
    if epoch_rows:
        path = os.path.join(results_sub, f"{model_key}_training_epoch_metrics.csv")
        pd.DataFrame(epoch_rows).to_csv(path, index=False)
        print(f"Training epoch metrics -> {path}")

def _save_zero_shot_results(zs_results, all_res, config, results_sub, model_key):
    """Save zero-shot CSV + zero-vs-LoRA ladder CSV + 2-panel ladder plot
    (Macro-F1 and Sarcasm-F1)."""
    zs_rows = [{k: v for k, v in r.items()
                if k not in ('confusion_matrix', 'classification_report', 'y_true', 'y_pred')}
               for r in zs_results]
    path_zs = os.path.join(results_sub, f"{model_key}_zero_shot_results.csv")
    pd.DataFrame(zs_rows).to_csv(path_zs, index=False)
    print(f"Zero-shot results      -> {path_zs}")

    varieties = config["varieties"]
    has_combined = any(r["variety_train"] == "combined" for r in all_res)
    ladder_varieties = varieties + (["combined"] if has_combined else [])

    def _get_sarc(r):
        return r.get("f1_sarcasm", r.get("f1_sarcastic", 0.0)) if r is not None else None

    ladder_rows = []
    for v in ladder_varieties:
        zs   = next((r for r in zs_results if r["variety_test"] == v), None)
        pv   = next((r for r in all_res
                     if r["variety_train"] == v and r["variety_test"] == v), None)
        comb = next((r for r in all_res
                     if r["variety_train"] == "combined" and r["variety_test"] == v), None)

        zs_f1   = zs["macro_f1"]   if zs   is not None else None
        pv_f1   = pv["macro_f1"]   if pv   is not None else None
        comb_f1 = comb["macro_f1"] if comb is not None else None

        zs_sarc   = _get_sarc(zs)
        pv_sarc   = _get_sarc(pv)
        comb_sarc = _get_sarc(comb)

        if zs_f1 is not None and comb_f1 is not None:
            ladder_rows.append({
                "test_variety":           v,
                # Macro-F1 columns
                "zero_shot_f1":           zs_f1,
                "in_dist_lora_f1":        pv_f1 if pv_f1 is not None else float("nan"),
                "combined_f1":            comb_f1,
                "lora_gain_over_zs":      round(pv_f1 - zs_f1, 4) if pv_f1 is not None else float("nan"),
                "combined_gain_over_zs":  round(comb_f1 - zs_f1, 4),
                # Sarcasm-F1 columns (merged in from pv_vs_combined)
                "zero_shot_f1_sarc":      zs_sarc   if zs_sarc   is not None else float("nan"),
                "in_dist_lora_f1_sarc":   pv_sarc   if pv_sarc   is not None else float("nan"),
                "combined_f1_sarc":       comb_sarc if comb_sarc is not None else float("nan"),
                # Per-variety vs combined delta (from pv_vs_combined)
                "delta_macro_f1_comb_vs_pv":
                    round(comb_f1 - pv_f1, 4) if pv_f1 is not None else float("nan"),
                "delta_sarc_f1_comb_vs_pv":
                    round(comb_sarc - pv_sarc, 4)
                    if (comb_sarc is not None and pv_sarc is not None) else float("nan"),
            })

    if not ladder_rows:
        return

    path_ladder = os.path.join(results_sub, f"{model_key}_zero_vs_lora_ladder.csv")
    pd.DataFrame(ladder_rows).to_csv(path_ladder, index=False)
    print(f"Zero vs LoRA ladder    -> {path_ladder}")

    df_ladder = pd.DataFrame(ladder_rows)

    # Mode-aware labels
    mode_tag       = "LoRA" if config.get("training_mode") == "lora" else "Full-FT"
    indist_label   = f"In-Dist {mode_tag}"
    combined_label = f"Combined {mode_tag}"

    # 2-panel figure: Macro-F1 (left) and Sarcasm-F1 (right)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f'{model_key} Macro-F1 & Sarcasm-F1 Ladder: '
        f'Zero-Shot -> In-Dist {mode_tag} -> Combined {mode_tag}',
        fontsize=12, fontweight='bold'
    )

    x = np.arange(len(df_ladder))
    w = 0.25

    for ax, (zs_col, pv_col, co_col, title) in zip(axes, [
        ("zero_shot_f1",      "in_dist_lora_f1",      "combined_f1",      "Macro-F1"),
        ("zero_shot_f1_sarc", "in_dist_lora_f1_sarc", "combined_f1_sarc", "Sarcasm-F1"),
    ]):
        b1 = ax.bar(x - w, df_ladder[zs_col], w,
                    label="Zero-Shot", color="#A9A9A9",
                    edgecolor="black", linewidth=0.5)
        b2 = ax.bar(x,     df_ladder[pv_col], w,
                    label=indist_label, color="#5B9BD5",
                    edgecolor="black", linewidth=0.5)
        b3 = ax.bar(x + w, df_ladder[co_col], w,
                    label=combined_label, color="#ED7D31",
                    edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(df_ladder["test_variety"], fontsize=11)
        ax.set_ylabel(f"{title} Score")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        for bar_set in [b1, b2, b3]:
            for bar in bar_set:
                h = bar.get_height()
                if h > 0 and not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path_plot = os.path.join(results_sub, f"{model_key}_zero_vs_lora_ladder.png")
    plt.savefig(path_plot, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Zero vs LoRA plot      -> {path_plot}")