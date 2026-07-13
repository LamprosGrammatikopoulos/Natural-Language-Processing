# Section 2.1 - Classical vs Transformer Baselines
# Member 2: Nikolas Ziartis (URN: 6911106) | Group 5
# NLP COMM061, University of Surrey, 2025-26

import argparse
import os
import torch
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ── Model Registry ──────────────────────────────────────────────────────────
# Maps architecture, task, and variety to the best saved model directory.
# Best seeds verified from pickle files across all training runs.
#
# Legend:
#   (a) Deterministic classical model — every seed gives identical results.
#       Trained on pooled data (all three varieties mixed).
#   (b) Seed-sensitive model trained on pooled data. Listed seed is the
#       best-performing of 3 runs by Macro-F1.
#   (c) Per-variety model: trained only on the single variety named in
#       the filename. Built in Q2.2.
#
# Only RoBERTa Sentiment has per-variety weights. All other slots use
# the same pooled file across all three variety folders — this is the
# correct intended behaviour.

MODEL_REGISTRY = {
    "classical": {
        "sentiment": {                                          # (a) F1=0.8407
            "combined": "classical_sentiment_seed42",
        },
        "sarcasm": {                                            # (a) F1=0.6275
            "combined": "classical_sarcasm_seed42",
        },
    },
    "fasttext": {
        "sentiment": {                                          # (b) F1=0.8310
            "combined": "fasttext_sentiment_seed123",
        },
        "sarcasm": {                                            # (b) F1=0.6200
            "combined": "fasttext_sarcasm_seed42",
        },
    },
    "roberta": {
        "sentiment": {
            "en-AU": "roberta_cv_en-AU_sentiment_seed123",      # (c) F1=0.8964
            "en-IN": "roberta_cv_en-IN_sentiment_seed42",       # (c) F1=0.8439
            "en-UK": "roberta_cv_en-UK_sentiment_seed42",       # (c) F1=0.9499
        },
        "sarcasm": {                                            # (b) F1=0.7124
            "combined": "roberta_sarcasm_seed456",
        },
    },
}

# Best classical classifier per task
BEST_CLASSICAL = {
    "sentiment": "svm",    # LinearSVM:          F1=0.8407
    "sarcasm": "lr",       # LogisticRegression:  F1=0.6275
}

LABEL_MAPS = {
    "sentiment": {0: "Negative", 1: "Positive"},
    "sarcasm": {0: "Non-Sarcastic", 1: "Sarcastic"},
}


# ── Classical ───────────────────────────────────────────────────────────────

def load_classical_and_predict(model_dir, text, task, variety="unknown"):
    """
    Loads TF-IDF + LogisticRegression + LinearSVM from joblib files.

    Model structure:
        model_dir/
        ├── tfidf.joblib    (max_features=50k, ngram=(1,2), sublinear_tf)
        ├── lr.joblib       (LogisticRegression, class_weight='balanced')
        └── svm.joblib      (LinearSVC, class_weight='balanced')
    """
    import joblib

    print(f"\nArchitecture : Classical (TF-IDF)")
    print(f"Task         : {task}")
    print(f"Directory    : {model_dir}")
    print(f"Variety      : {variety}")

    tfidf = joblib.load(os.path.join(model_dir, "tfidf.joblib"))
    lr = joblib.load(os.path.join(model_dir, "lr.joblib"))
    svm = joblib.load(os.path.join(model_dir, "svm.joblib"))

    X = tfidf.transform([text])

    lr_pred = int(lr.predict(X)[0])
    svm_pred = int(svm.predict(X)[0])

    best_clf_name = BEST_CLASSICAL[task]
    if best_clf_name == "svm":
        pred_class = svm_pred
        decision = float(svm.decision_function(X)[0])
        confidence = 1.0 / (1.0 + np.exp(-abs(decision)))
        clf_label = "LinearSVM"
    else:
        pred_class = lr_pred
        proba = lr.predict_proba(X)[0]
        confidence = float(max(proba))
        clf_label = "LogisticRegression"

    labels = LABEL_MAPS[task]

    print("\n" + "=" * 55)
    print(f"  Text       : {text}")
    print(f"  Variety    : {variety}")
    print(f"  Task       : {task}")
    print(f"  Classifier : {clf_label} (best for {task})")
    print(f"  Prediction : {labels[pred_class]}")
    print(f"  Confidence : {confidence:.1%}")
    print(f"  LR pred    : {labels[lr_pred]}")
    print(f"  SVM pred   : {labels[svm_pred]}")
    print("=" * 55)

    return pred_class, confidence


# ── FastText ────────────────────────────────────────────────────────────────

def load_fasttext_and_predict(model_dir, text, task, variety="unknown"):
    """
    Loads a FastText supervised model from a .bin file.

    Model structure:
        model_dir/
        └── model.bin   (FastText supervised, autotuned 120s, oversampled minority)
    """
    import fasttext
    fasttext.FastText.eprint = lambda x: None  # suppress warnings

    print(f"\nArchitecture : FastText (supervised)")
    print(f"Task         : {task}")
    print(f"Directory    : {model_dir}")
    print(f"Variety      : {variety}")

    model_file = os.path.join(model_dir, "model.bin")
    model = fasttext.load_model(model_file)

    clean = ' '.join(str(text).replace('\n', ' ').split())

    try:
        pred_labels, pred_probs = model.predict(clean)
    except ValueError:
        # NumPy 2.x workaround
        pairs = model.f.predict(clean, 1, 0.0, 'strict')
        if pairs:
            pred_probs, pred_labels = zip(*pairs)
        else:
            pred_labels = ('__label__0',)
            pred_probs = (1.0,)

    pred_class = int(pred_labels[0].replace('__label__', ''))
    confidence = float(pred_probs[0])

    labels = LABEL_MAPS[task]

    print("\n" + "=" * 55)
    print(f"  Text       : {text}")
    print(f"  Variety    : {variety}")
    print(f"  Task       : {task}")
    print(f"  Prediction : {labels[pred_class]}")
    print(f"  Confidence : {confidence:.1%}")
    print("=" * 55)

    return pred_class, confidence


# ── RoBERTa ─────────────────────────────────────────────────────────────────

def load_roberta_and_predict(model_dir, text, task, variety="unknown"):
    """
    Loads fine-tuned RoBERTa-base from HuggingFace save format.

    Model structure:
        model_dir/
        ├── config.json
        ├── model.safetensors (or pytorch_model.bin)
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── vocab.json
        ├── merges.txt
        └── special_tokens_map.json

    Training: roberta-base, lr=2e-5, batch=16, max 5 epochs,
    early stopping (patience=2), class_weight='balanced'.
    """
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()

    print(f"\nArchitecture : RoBERTa-base (fine-tuned)")
    print(f"Task         : {task}")
    print(f"Directory    : {model_dir}")
    print(f"Variety      : {variety}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device       : CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device       : CPU")

    print("\nLoading tokenizer ...")
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)

    print("Loading RoBERTa model ...")
    model = RobertaForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    inputs = tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_class = torch.argmax(logits, dim=-1).item()

        probs = torch.softmax(logits, dim=-1).squeeze()
        prob_0 = probs[0].item()
        prob_1 = probs[1].item()

    labels = LABEL_MAPS[task]

    print("\n" + "=" * 55)
    print(f"  Text       : {text}")
    print(f"  Variety    : {variety}")
    print(f"  Task       : {task}")
    print(f"  Prediction : {labels[pred_class]}")
    print(f"  Confidence : {max(prob_0, prob_1):.1%}")
    print(f"  P({labels[0]})  = {prob_0:.4f}")
    print(f"  P({labels[1]}) = {prob_1:.4f}")
    print("=" * 55)

    return pred_class, max(prob_0, prob_1)


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Sentiment and Sarcasm inference using Classical (TF-IDF), "
            "FastText, and RoBERTa baselines from Q2.1."
        )
    )

    parser.add_argument(
        "--arch",
        type=str,
        choices=["classical", "fasttext", "roberta", "all"],
        default="all",
        help="Which model architecture to use. 'all' runs all three. Default: all",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["sentiment", "sarcasm", "both"],
        default="both",
        help="Which task to run. 'both' runs sentiment and sarcasm. Default: both",
    )

    parser.add_argument(
        "--variety",
        type=str,
        choices=["en-UK", "en-AU", "en-IN"],
        default="en-UK",
        help="Which variety model to use. Default: en-UK",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Optional: override the model path directly.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="Oh brilliant, it's raining again.",
        help="The text snippet to classify.",
    )

    args = parser.parse_args()

    import gc
    script_dir = os.path.dirname(os.path.abspath(__file__))

    archs_to_run = (
        ["classical", "fasttext", "roberta"] if args.arch == "all" else [args.arch]
    )
    tasks_to_run = (
        ["sentiment", "sarcasm"] if args.task == "both" else [args.task]
    )

    PREDICT_FN = {
        "classical": load_classical_and_predict,
        "fasttext": load_fasttext_and_predict,
        "roberta": load_roberta_and_predict,
    }

    print(f"\nEvaluating text: '{args.text}'")
    print(f"Variety: {args.variety}")
    print("=" * 60)

    results = {}

    for arch in archs_to_run:
        for task in tasks_to_run:
            key = f"{arch}_{task}"

            # Resolve path
            if args.model_dir is not None:
                model_dir = args.model_dir
            else:
                raw_dir = MODEL_REGISTRY[arch][task][args.variety]
                model_dir = os.path.join(script_dir, "saved_models", raw_dir)

            if not os.path.exists(model_dir):
                print(f"\n⚠ Skipping {key} — {model_dir} not found")
                continue

            pred, conf = PREDICT_FN[arch](model_dir, args.text, task, args.variety)
            results[key] = {"pred": pred, "confidence": conf, "task": task}

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for key, res in results.items():
            task = res["task"]
            labels = LABEL_MAPS[task]
            label = labels[res["pred"]]
            print(f"{key:<28} : {label:<15} (Confidence: {res['confidence']:.1%})")
        print("=" * 60 + "\n")
