import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from peft import PeftModel
from pathlib import Path

# ==========================================
# 1. SETUP & LOADING (Runs once at startup)
# ==========================================

ADAPTER_REGISTRY = {
    "deberta": {
        "sarcasm": {
            "combined": "encoder_variations_2_2/outputs_deberta/sarcasm/checkpoints/pooled_seed123",
            "en-AU": "deberta_adapters/sarcasm/en-AU/deberta_combined_seed123",
            "en-IN": "deberta_adapters/sarcasm/en-IN/deberta_cross_train-en-IN_seed456",
            "en-UK": "deberta_adapters/sarcasm/en-UK/deberta_combined_seed123"
        },
        "sentiment": {
            "combined": "deberta_adapters/sentiment/combined/deberta_pooled_seed123",
            "en-AU": "deberta_adapters/sentiment/en-AU/deberta_combined_seed123",
            "en-IN": "deberta_adapters/sentiment/en-IN/deberta_combined_seed123",
            "en-UK": "deberta_adapters/sentiment/en-UK/deberta_combined_seed123",
        }
    },
    "rembert": {
        "sarcasm": {
            "combined": "rembert_adapters/sarcasm/combined/rembert_pooled_seed456",
            "en-AU": "rembert_adapters/sarcasm/en-AU/rembert_combined_seed123",
            "en-IN": "rembert_adapters/sarcasm/en-IN/rembert_cross_train-en-IN_seed456",
            "en-UK": "rembert_adapters/sarcasm/en-UK/rembert_cross_train-en-UK_seed42",
        },
        "sentiment": {
            "combined": "rembert_adapters/sentiment/combined/rembert_pooled_seed456",
            "en-AU": "rembert_adapters/sentiment/en-AU/rembert_combined_seed456",
            "en-IN": "rembert_adapters/sentiment/en-IN/rembert_cross_train-en-IN_seed456",
            "en-UK": "rembert_adapters/sentiment/en-UK/rembert_cross_train-en-UK_seed456",
        }
    },
    "xlm_roberta": {
        "sentiment": {
            "combined": "xlm_roberta_adapters/sentiment/combined/xlmroberta_sentiment_pooled_seed123",
            "en-AU": "xlm_roberta_adapters/sentiment/en-AU/xlmroberta_sentiment_cross_train-en-AU_seed456",
            "en-IN": "xlm_roberta_adapters/sentiment/en-IN/xlmroberta_sentiment_combined_seed42",
            "en-UK": "xlm_roberta_adapters/sentiment/en-UK/xlmroberta_sentiment_combined_seed42",
        }
    }
}

BASE_MODEL_NAME = {
    "deberta": "microsoft/deberta-v3-base",
    "rembert": "google/rembert",
    "xlm_roberta": "xlm-roberta-base",
}


def load_peft_and_predict(arch, adapter_path, text, task, variety):
    """
    Loads the LoRA adapter onto the base model and runs a single prediction.
    """

    base_model_name = BASE_MODEL_NAME[arch]

    print("\n" + "=" * 55)
    print(f"Architecture : {arch}")
    print(f"Task         : {task}")
    print(f"Directory    : {adapter_path}")
    print(f"Variety      : {variety}")
    print()

    print("Loading tokenizer ...")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    print("Loading adapter weights...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Using pipeline for inference
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)

    result = classifier(text)[0]

    # Map LABEL_1 to Positive
    pred_class = 1 if result["label"] == "LABEL_1" else 0
    confidence = result["score"]

    labels = (
        ["Not Sarcastic", "Sarcastic"]
        if task == "sarcasm"
        else ["Negative", "Positive"]
    )

    pred_label = labels[pred_class]

    print()
    print("=" * 55)
    print(f"Text       : {text}")
    print(f"Variety    : {variety}")
    print(f"Task       : {task}")
    print(f"Prediction : {pred_label}")
    print(f"Confidence : {confidence:.1%}")
    print("=" * 55)

    return pred_class, confidence

def predict_sarcasm_2_2(arch, variety_key, text, ui_map):
    try:
        variety = ui_map[variety_key]
        script_dir = Path(__file__).resolve().parent.parent
        base_path = script_dir / "encoder_variations_2_2" / "saved_models"

        start_time = time.perf_counter()
        results = {}


        tasks = ["sentiment"] if arch == "xlm_roberta" else ["sarcasm", "sentiment"]


        for task in tasks:            
            raw_dir = ADAPTER_REGISTRY[arch][task][variety]
            adapter_path = base_path / raw_dir

            pred_class, confidence = load_peft_and_predict(arch, str(adapter_path), text, task, variety)

            # Map labels
            if arch == "xlm_roberta":
                labels = ["Not Sarcastic", "Sarcastic"]
            else:
                labels = ["Not Sarcastic", "Sarcastic"] if task == "sarcasm" else ["Negative", "Positive"]
            results[task] = (labels[pred_class], confidence)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        if arch == "xlm_roberta":
            return(
                results["sentiment"][0],
                f"Sentiment: {results['sentiment'][1]:.1%}",
                f"Latency: {latency_ms:.2f} ms"
            )
        else:
            return (
                f"{results['sarcasm'][0]}  |  {results['sentiment'][0]}",
                f"Sarcasm: {results['sarcasm'][1]:.1%}  |  Sentiment: {results['sentiment'][1]:.1%}",
                f"Latency: {latency_ms:.2f} ms"
            )
    except Exception as e:
        return "Error", f"Inference failed: {str(e)}", "N/A"
