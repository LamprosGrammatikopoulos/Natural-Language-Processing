# Section 2.3 - LLM Variations with LoRA Adapters
# Member 4: Lampros Grammatikopoulos (URN: 6918674) | Group 5
# NLP COMM061, University of Surrey, 2025-26

import argparse
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
from peft import PeftModel

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# Adapter Registry
# Maps architecture and variety names to their best saved adapter/checkpoint directories.
ADAPTER_REGISTRY = {
    "tinyllama": {
        "en-UK": "tinyllama_adapters/en_UK_run1",
        "en-AU": "tinyllama_adapters/en_AU_run3",
        "en-IN": "tinyllama_adapters/en_IN_run3",
        "combined": "tinyllama_adapters/combined_run2"
    },
    "xlmr": {
        "en-UK": "xlmr_checkpoints/en_UK_run1",
        "en-AU": "xlmr_checkpoints/en_AU_run2",
        "en-IN": "xlmr_checkpoints/en_IN_run1",
        "combined": "xlmr_checkpoints/combined_run3"
    },
    "qwen25": {
        "en-UK": "qwen_adapters/en_UK_run2",
        "en-AU": "qwen_adapters/en_AU_run2",
        "en-IN": "qwen_adapters/en_IN_run1",
        "combined": "qwen_adapters/combined_run1"
    },
    "xlmr_lora": {
        "en-UK": "xlmr_lora_adapters/en_UK_run2",
        "en-AU": "xlmr_lora_adapters/en_AU_run3",
        "en-IN": "xlmr_lora_adapters/en_IN_run1",
        "combined": "xlmr_lora_adapters/combined_run1"
    },
}

BASE_MODELS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen25": "Qwen/Qwen2.5-1.5B",
    "xlmr": "xlm-roberta-base",
    "xlmr_lora": "xlm-roberta-base",   # LoRA overlays on the same base
}

# Architectures that use LoRA (base model + PeftModel adapter overlay).
# Anything not in this set is loaded directly as a full fine-tuned checkpoint.
LORA_ARCHS = {"tinyllama", "qwen25", "xlmr_lora"}

# Architectures that are decoder-only (causal LMs) and therefore need LEFT padding.
# Encoders (XLM-R, both full-FT and LoRA variants) use RIGHT padding.
CAUSAL_ARCHS = {"tinyllama", "qwen25"}


def load_model_and_predict(arch, model_dir, text, variety="unknown"):
    """
    Loads the model based on the architecture and predicts Sarcasm (1) or Not Sarcastic (0).

    Parameters
    ----------
    arch        : str   Architecture (tinyllama, qwen25, xlmr, xlmr_lora)
    model_dir   : str   Path to directory containing LoRA adapter or fine-tuned checkpoint
    text        : str   Input text to classify
    variety     : str   Variety label for display purposes
    """
    base_model_name = BASE_MODELS[arch]

    print(f"\nArchitecture : {arch}")
    print(f"Base Model   : {base_model_name}")
    print(f"Directory    : {model_dir}")
    print(f"Variety      : {variety}")

    # 1. Determine compute dtype
    if torch.cuda.is_available():
        major = torch.cuda.get_device_properties(0).major
        compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
        device_map = "auto"
        print(f"Device       : CUDA ({torch.cuda.get_device_name(0)})")
    else:
        compute_dtype = torch.float32
        device_map = None
        print("Device       : CPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Tokenizer
    print("\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Decoder-only models require left-padding for classification so that the
    # final hidden state corresponds to the real last token rather than a pad.
    # Encoders (XLM-R full-FT and XLM-R LoRA) use standard right-padding.
    if arch in CAUSAL_ARCHS:
        tokenizer.padding_side = "left"
    else:
        tokenizer.padding_side = "right"

    # 3. Load model
    print(f"Loading {arch} model ...")
    if arch in LORA_ARCHS:
        # LoRA path: load the frozen base, then overlay the PEFT adapter.
        # This is used for TinyLlama, Qwen2.5 AND XLM-R + LoRA.
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=2,
            torch_dtype=compute_dtype,
            device_map=device_map,
        )
        model.config.problem_type = "single_label_classification"
        print("Applying LoRA adapter ...")
        model = PeftModel.from_pretrained(model, model_dir)
    else:
        # Full fine-tuning path: load the saved checkpoint directly.
        # Currently only XLM-R (full FT) uses this branch.
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=2,
            torch_dtype=compute_dtype,
            device_map=device_map,
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # 4. Tokenize input
    inputs = tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 5. Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_class = torch.argmax(logits, dim=-1).item()

        # Softmax probabilities for both classes
        probs = torch.softmax(logits, dim=-1).squeeze()
        prob_not = probs[0].item()
        prob_sarc = probs[1].item()

    label_map = {0: "Not Sarcastic", 1: "Sarcastic"}

    print("\n" + "=" * 55)
    print(f"  Text       : {text}")
    print(f"  Variety    : {variety}")
    print(f"  Prediction : {label_map[pred_class]}")
    print(f"  Confidence : {max(prob_not, prob_sarc):.1%}")
    print(f"  P(Not-Sarc) = {prob_not:.4f}")
    print(f"  P(Sarcastic) = {prob_sarc:.4f}")
    print("=" * 55)

    return pred_class, prob_sarc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sarcasm Detection inference supporting TinyLlama, Qwen2.5, XLM-RoBERTa (full FT) and XLM-RoBERTa + LoRA."
    )

    parser.add_argument(
        "--arch",
        type=str,
        choices=["tinyllama", "qwen25", "xlmr", "xlmr_lora", "all"],
        default="all",
        help="Which model architecture to use. 'all' evaluates and compares all four. Default: all",
    )

    parser.add_argument(
        "--variety",
        type=str,
        choices=["en-UK", "en-AU", "en-IN", "combined"],
        default="combined",
        help=(
            "Which variety model to use: "
            "en-AU (Australian), en-UK (British), en-IN (Indian), "
            "or combined (all varieties pooled). Default: combined"
        ),
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help=(
            "Optional: override the path directly. "
            "If set, --variety is used for display only."
        ),
    )

    parser.add_argument(
        "--text",
        type=str,
        default="Oh brilliant, it's raining again.",
        help="The text snippet to classify for sarcasm.",
    )

    args = parser.parse_args()

    import os
    import gc
    script_dir = os.path.dirname(os.path.abspath(__file__))

    archs_to_run = (
        ["tinyllama", "qwen25", "xlmr", "xlmr_lora"]
        if args.arch == "all"
        else [args.arch]
    )

    print(f"\nEvaluating text: '{args.text}'")
    print("=" * 60)

    results = {}
    for arch in archs_to_run:
        # Resolve path
        if args.model_dir is not None:
            raw_dir = args.model_dir
        else:
            raw_dir = ADAPTER_REGISTRY[arch][args.variety]

        model_dir = os.path.join(script_dir, raw_dir) if not os.path.isabs(raw_dir) else raw_dir

        pred_class, prob_sarc = load_model_and_predict(arch, model_dir, args.text, args.variety)
        results[arch] = {"pred": pred_class, "prob": prob_sarc}

        # Clean up memory between giant model loads
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.arch == "all":
        print("\n" + "=" * 60)
        print("FINAL ENSEMBLE SUMMARY")
        print("=" * 60)
        for arch, res in results.items():
            label = "Sarcastic" if res["pred"] == 1 else "Not Sarcastic"
            print(f"{arch.upper():<12} : {label:<15} (Confidence: {max(res['prob'], 1 - res['prob']):.1%})")

        # Simple majority vote across all 4 models. Ties (2-2) default to Not Sarcastic.
        n_models = len(results)
        sarc_votes = sum(1 for res in results.values() if res["pred"] == 1)
        majority_label = "Sarcastic" if sarc_votes > n_models / 2 else "Not Sarcastic"
        print("-" * 60)
        print(f"Majority Vote: {majority_label} ({sarc_votes}/{n_models} models)")
        print("=" * 60 + "\n")