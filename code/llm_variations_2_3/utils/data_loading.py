"""
utils/data_loading.py - Dataset loading, filtering, balancing.

Usage:
    from utils.data_loading import load_and_prepare_data
    variety_data, raw_datasets = load_and_prepare_data(CONFIG)
"""
import os
from collections import Counter
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict

# Helpers 
VARIETY_VALUE_MAP = {
    "en-AU": ["en-AU", "en_AU", "en-au", "au", "australian"],
    "en-UK": ["en-UK", "en_UK", "en-uk", "uk", "british", "en-GB", "en_GB"],
    "en-IN": ["en-IN", "en_IN", "en-in", "in", "indian"],
}

def _detect_columns(sample_ds):
    """Auto-detect text, label, and variety columns."""
    cols = sample_ds.column_names
    text_col = next((c for c in ["text", "sentence", "review", "comment", "utterance", "Text"] if c in cols), cols[0])
    label_col = next((c for c in ["Sarcasm", "sarcasm", "sarcasm_label", "sarcastic", "label", "Sarcastic"] if c in cols), None)
    if label_col is None:
        raise ValueError("Cannot auto-detect sarcasm label column. Available: {cols}")

    variety_col = next((c for c in cols if any(kw in c.lower() for kw in ["variety", "lang", "locale", "region", "dialect", "country"])), None)
    if variety_col is None:
        raise ValueError("Cannot auto-detect variety column. Available: {cols}")

    return text_col, label_col, variety_col

def _prepare_variety(ds, variety, text_col, label_col, max_train=None, seed=42):
    """Balance train split, leave val/test untouched."""

    def to_df(split):
        if split not in ds:
            return None
        df = pd.DataFrame({"text": ds[split][text_col], "label": ds[split][label_col]})
        df["label"] = df["label"].astype(int)
        return df

    train_df = to_df("train")
    val_df   = to_df("validation")
    test_df  = to_df("test")

    # Optional cap on training size
    if max_train and train_df is not None and len(train_df) > max_train:
        minority  = train_df[train_df.label == 1]
        majority  = train_df[train_df.label == 0]
        n_maj     = max(min(len(majority), max_train - len(minority)), 1)
        train_df  = pd.concat([minority, majority.sample(n_maj, random_state=seed)])
        train_df  = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"  [{variety}] Training limited to {len(train_df)} samples")

    # Oversample minority class
    if train_df is not None:
        minority = train_df[train_df.label == 1]
        majority = train_df[train_df.label == 0]
        if len(minority) > 0 and len(majority) > len(minority):
            n_extra  = len(majority) - len(minority)
            extra    = minority.sample(n_extra, replace=True, random_state=seed)
            train_df = pd.concat([majority, minority, extra])
            train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Summary
    for name, df in [("Train (balanced)", train_df),("Val", val_df), ("Test", test_df)]:
        if df is not None:
            c = Counter(df.label)
            print(f"  [{variety}] {name:<20}: N={len(df)}  Sarc={c[1]}  Non-Sarc={c[0]}")

    return train_df, val_df, test_df

def _build_combined(variety_data, varieties, seed=42):
    """Pool all varieties into a combined dataset."""
    all_tr, all_va, all_te = [], [], []
    for v in varieties:
        if v not in variety_data:
            continue
        for split_key, lst in [("train", all_tr),("val", all_va), ("test", all_te)]:
            df = variety_data[v][split_key].copy()
            df["label"] = df["label"].astype(int)
            df["variety_origin"] = v
            lst.append(df)

    def _merge(frames):
        return (pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True))

    comb_tr = _merge(all_tr)
    comb_va = _merge(all_va)
    comb_te = _merge(all_te)

    print("Combined dataset overview:")
    print("-" * 55)
    for name, df in [("Train", comb_tr), ("Val", comb_va)]:
        c = Counter(df["label"].astype(int))
        total = len(df)
        print(f"  {name:<6}: {total:>5} samples  Sarcastic={c[1]:>4}  Non-Sarcastic={c[0]:>4}  Sarc%={c[1]/total:.1%}")
    print("\n  Per-variety contribution to combined training split:")
    for v, cnt in comb_tr["variety_origin"].value_counts().sort_index().items():
        print(f"    {v}: {cnt} rows ({cnt/len(comb_tr):.1%})")

    return {"train": comb_tr, "val": comb_va, "test": comb_te}


def _plot_class_distribution(variety_data, varieties, results_dir):
    """Bar chart of class distributions per variety."""
    os.makedirs(os.path.join(results_dir, "data"), exist_ok=True)

    fig, axes = plt.subplots(1, len(varieties), figsize=(14, 4))
    if len(varieties) == 1:
        axes = [axes]
    fig.suptitle("Sarcasm Class Distribution – Training Split (after oversampling)", fontsize=13, fontweight="bold")
    palette = {"Non-Sarcastic (0)": "#5B9BD5", "Sarcastic (1)": "#ED7D31"}

    for ax, variety in zip(axes, varieties):
        if variety not in variety_data:
            ax.set_visible(False)
            continue
        df = variety_data[variety]["train"]
        c  = Counter(df.label)
        labels_x = ["Non-Sarcastic (0)", "Sarcastic (1)"]
        counts   = [c.get(0, 0), c.get(1, 0)]
        bars = ax.bar(labels_x, counts, color=[palette[l] for l in labels_x], edgecolor="black", linewidth=0.6)
        ax.set_title(variety, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(counts) * 1.25)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.02,
                    str(count), ha="center", va="bottom",
                    fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "data", "class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Class-distribution plot saved")

# Public API
def load_and_prepare_data(config: dict):
    """
    Load the BESSTIE dataset, filter by variety, balance, build
    combined split, and plot distributions.

    Returns
    -------
    variety_data  : dict[str, dict]   Per-variety DataFrames
                    Keys: variety codes + 'combined'
                    Sub-keys: 'train', 'val', 'test'
    raw_datasets  : dict[str, DatasetDict]
    """
    dataset_name = config["dataset_name"]
    varieties    = config["varieties"]
    seed         = config["seed"]

    print(f"Downloading dataset from HuggingFace Hub: {dataset_name}")
    full_ds = load_dataset(dataset_name)
    print(f"  Splits   : {list(full_ds.keys())}")
    print(f"  Columns  : {full_ds['train'].column_names}")
    print(f"  Total train rows: {len(full_ds['train'])}")

    text_col, label_col, variety_col = _detect_columns(full_ds["train"])
    print(f"\n  Text column  : '{text_col}'")
    print(f"  Label column : '{label_col}'")
    print(f"  Variety col  : '{variety_col}'")

    # Filter each variety 
    raw_datasets = {}
    for variety in varieties:
        aliases = [a.lower() for a in VARIETY_VALUE_MAP.get(variety, [variety])]
        filtered = DatasetDict({
            split: full_ds[split].filter(lambda row, a=aliases, vc=variety_col: str(row[vc]).lower() in a)
            for split in full_ds.keys()
        })
        if len(filtered.get("train", [])) == 0:
            print(f"  !! {variety}: 0 rows found - skipping")
            continue
        raw_datasets[variety] = filtered
        for split, ds in filtered.items():
            print(f"  {variety} | {split:<12}: {len(ds):>5} rows")
        print()

    # Prepare & balance 
    print("Preparing data for all varieties:")
    variety_data = {}
    for v in varieties:
        if v not in raw_datasets:
            continue
        tr, va, te = _prepare_variety(
            raw_datasets[v], v, text_col, label_col,
            max_train=config["max_train_samples"], seed=seed,
        )
        variety_data[v] = {"train": tr, "val": va, "test": te}
        print()

    # Combined dataset 
    combined = _build_combined(variety_data, varieties, seed)
    variety_data["combined"] = combined

    # Distribution plot 
    _plot_class_distribution(variety_data, varieties, config["results_dir"])

    # Store detected column names in config for downstream use
    config["text_col"]  = text_col
    config["label_col"] = label_col

    return variety_data, raw_datasets
