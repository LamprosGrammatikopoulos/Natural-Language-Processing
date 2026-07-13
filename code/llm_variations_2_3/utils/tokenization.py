"""
utils/tokenization.py - Tokenizer loading + dataset tokenization with disk-based caching.
Usage:
    from utils.tokenization import load_tokenizer_and_tokenize
    tokenizer, tokenized = load_tokenizer_and_tokenize(CONFIG, variety_data)
"""
import os, torch
from datasets import Dataset as HFDataset, load_from_disk
from transformers import AutoTokenizer

# Tokenize a single DataFrame 
def _tokenize_df(df, tokenizer_obj, config):
    """Convert a pandas DataFrame -> tokenized HF Dataset."""
    text_col  = config.get("text_col", "text")
    label_col = config.get("label_col", "Sarcasm")

    # Fallback column detection
    if label_col not in df.columns:
        for cand in ["Sarcasm", "sarcasm", "label"]:
            if cand in df.columns:
                label_col = cand
                break
        else:
            raise KeyError(f"Cannot find label column. Available: {df.columns.tolist()}")

    hf = HFDataset.from_pandas(df[[text_col, label_col]].reset_index(drop=True))

    def _tok(batch, tok=tokenizer_obj, t_col=text_col):
        return tok(
            batch[t_col],
            max_length  = config["max_length"],
            truncation  = True,
            padding     = False,
        )

    hf = hf.map(_tok, batched=True, remove_columns=[text_col])
    hf = hf.rename_column(label_col, "labels")
    hf.set_format("torch")
    return hf

# Hard-cast labels to int64 
def _recast_labels(tokenized):
    """Ensure all label tensors are Long (int64)."""
    for variety in tokenized:
        for split in tokenized[variety]:
            ds = tokenized[variety][split]
            new_labels = [int(float(x)) for x in ds["labels"]]
            ds = ds.remove_columns(["labels"])
            ds = ds.add_column("labels", new_labels)
            tokenized[variety][split] = ds.with_format("torch")
    print("✓ Dataset labels hard-cast to Long (int64).")

# Public API 
def load_tokenizer_and_tokenize(config, variety_data):
    """
    Load the tokenizer for ``config["base_model"]``, set up padding,
    then tokenize all variety splits with disk caching.

    Returns
    -------
    tokenizer  : PreTrainedTokenizer
    tokenized  : dict[str, dict[str, HFDataset]]
    """
    base_model = config["base_model"]
    cache_dir  = config["cache_dir"]

    # Load tokenizer 
    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Causal (decoder-only) LLMs often lack a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"  pad_token set to eos_token: '{tokenizer.eos_token}'")

    # Decoder-only models use left-padding for classification
    if config.get("is_causal_lm", False):
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"

    print(f"  Vocab size    : {tokenizer.vocab_size:,}")
    print(f"  Max length    : {config['max_length']} tokens (truncated)")
    print(f"  Padding side  : {tokenizer.padding_side}")
    print("Tokenizer ready\n")

    # Tokenize all varieties + combined 
    tokenized = {}
    for v, splits in variety_data.items():
        print(f"Processing {v} ...")
        tokenized[v] = {}
        for split_name, df in splits.items():
            if df is None or len(df) == 0:
                continue

            # Strip helper columns before tokenizing
            df_clean = df.drop(columns=["variety_origin"], errors="ignore")

            dataset_path = os.path.join(cache_dir, f"{v}_{split_name}")

            if os.path.exists(dataset_path):
                tokenized[v][split_name] = load_from_disk(dataset_path)
                n = len(tokenized[v][split_name])
                print(f"  {split_name:<12}: Loaded from disk "
                      f"({n} samples)")
            else:
                tokenized[v][split_name] = _tokenize_df(
                    df_clean, tokenizer, config
                )
                tokenized[v][split_name].save_to_disk(dataset_path)
                n = len(tokenized[v][split_name])
                print(f"  {split_name:<12}: Tokenized & saved "
                      f"({n} samples)")

    print("\nProcessing complete")

    # Re-cast labels 
    _recast_labels(tokenized)

    return tokenizer, tokenized