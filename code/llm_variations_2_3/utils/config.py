"""
utils/config.py - Architecture-specific configuration.

Usage:
    from utils.config import get_config
    CONFIG = get_config("tinyllama")   # or "qwen25", "xlmr", "xlmr_lora"
"""
import os, torch

# GPU precision detection 
def get_compute_dtype():
    """Return (dtype, use_fp16, use_bf16) based on GPU capability."""
    if not torch.cuda.is_available():
        return torch.float32, False, False
    major = torch.cuda.get_device_properties(0).major
    if major >= 8:          # Ampere+ -> BF16
        return torch.bfloat16, False, True
    else:                   # Turing / Volta -> FP16
        return torch.float16, True, False

# Shared defaults 
_SHARED = dict(
    dataset_name      = "surrey-nlp/BESSTIE-CW-26",
    varieties         = ["en-AU", "en-UK", "en-IN"],
    max_train_samples = 800,
    num_epochs        = 3,
    batch_size        = 8,
    learning_rate     = 2e-4,
    weight_decay      = 0.01,
    max_length        = 128,
    num_runs          = 3,
    results_dir       = "./results",
    seed              = 42,
)

# Per-architecture overrides 
_CONFIGS = {
    "tinyllama_lora": dict(
        model_key       = "tinyllama",
        base_model      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        training_mode   = "lora",          # "lora" | "full_finetune"
        is_causal_lm    = True,            # decoder-only -> left-pad
        lora_r          = 8,
        lora_alpha      = 16,
        lora_dropout    = 0.05,
        target_modules  = ["q_proj", "v_proj"],
        output_dir      = "./tinyllama_adapters",
        cache_dir       = "cached_tokenized_datasets_tinyllama",
        results_subdir  = "tinyllama",
    ),
    "qwen25_lora": dict(
        model_key       = "qwen25",
        base_model      = "Qwen/Qwen2.5-1.5B",
        training_mode   = "lora",
        is_causal_lm    = True,
        lora_r          = 8,
        lora_alpha      = 16,
        lora_dropout    = 0.05,
        target_modules  = ["q_proj", "v_proj"],
        output_dir      = "./qwen_adapters",
        cache_dir       = "cached_tokenized_datasets_qwen",
        results_subdir  = "qwen25",
    ),
    "xlmr": dict(
        model_key       = "xlmr",
        base_model      = "xlm-roberta-base",
        training_mode   = "full_finetune",   # full fine-tuning, NOT LoRA
        is_causal_lm    = False,             # encoder -> right-pad
        output_dir      = "./xlmr_checkpoints",
        cache_dir       = "cached_tokenized_datasets_xlmr",
        results_subdir  = "xlmr",
    ),
    "xlmr_lora": dict(
        model_key       = "xlmr_lora",
        base_model      = "xlm-roberta-base",
        training_mode   = "lora",
        is_causal_lm    = False,
        lora_r          = 8,
        lora_alpha      = 16,
        lora_dropout    = 0.05,
        target_modules  = ["query", "value"],   # RoBERTa attention names
        output_dir      = "./xlmr_lora_adapters",
        cache_dir       = "cached_tokenized_datasets_xlmr",  # reuse same cache
        results_subdir  = "xlmr_lora",
    ),
}


def get_config(model_key: str) -> dict:
    """
    Build the full CONFIG dict for a given architecture.

    Parameters
    ----------
    model_key : one of "tinyllama", "qwen25", "xlmr", "xlmr_lora"

    Returns
    -------
    dict  Complete configuration ready for all pipeline stages.
    """
    if model_key not in _CONFIGS:
        raise ValueError(f"Unknown model_key '{model_key}'. Choose from: {list(_CONFIGS.keys())}")

    cfg = {**_SHARED, **_CONFIGS[model_key]}

    # Ensure output directories exist
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["results_dir"], cfg["results_subdir"]), exist_ok=True)

    # Attach GPU precision info
    compute_dtype, use_fp16, use_bf16 = get_compute_dtype()
    cfg["compute_dtype"] = compute_dtype
    cfg["use_fp16"]      = use_fp16
    cfg["use_bf16"]      = use_bf16

    return cfg
