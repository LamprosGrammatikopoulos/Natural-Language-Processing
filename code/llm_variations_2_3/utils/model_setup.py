"""
utils/model_setup.py - Base model loading + LoRA / full-FT configuration.

Usage:
    from utils.model_setup import load_base_model, get_lora_config
    base_model, total_params = load_base_model(CONFIG, tokenizer)
    lora_cfg = get_lora_config(CONFIG)       # None for full_finetune
"""
import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model

def load_base_model(config, tokenizer):
    """
    Load the base model for sequence classification.
    Returns (model_on_cpu, total_params).
    """
    print(f"Loading base model: {config['base_model']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"],
        num_labels=2,
        torch_dtype=config["compute_dtype"],
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "single_label_classification"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Compute dtype   : {config['compute_dtype']}")

    # Keep on CPU - train_adapter will deep-copy per run
    model = model.cpu().eval()
    print("Base model ready on CPU\n")
    return model, total_params

def get_lora_config(config):
    """
    Return a LoraConfig if training_mode == 'lora', else None.
    Also previews trainable parameter count if LoRA is used.
    """
    if config["training_mode"] != "lora":
        print("Training mode: full fine-tuning (no LoRA)")
        return None

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        inference_mode=False,
    )

    print(f"LoRA configuration:")
    print(f"  Rank r          : {lora_cfg.r}")
    print(f"  Alpha           : {lora_cfg.lora_alpha}")
    print(f"  Target modules  : {lora_cfg.target_modules}")
    return lora_cfg
