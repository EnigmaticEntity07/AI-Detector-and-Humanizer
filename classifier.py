"""
classifier.py
=============
Inference module for the trained AI detector classifier.

Loads the persisted ``.pkl`` model bundle and exposes a simple ``predict()``
function that accepts raw text and returns a classification dict.

Usage
-----
    from classifier import predict

    result = predict("Some text to classify...")
    # result = {
    #     "label": 1,            # 0 = human, 1 = AI
    #     "probability": 0.87,   # P(AI)
    #     "threshold": 0.35,     # tuned decision threshold
    #     "verdict": "AI-Generated",
    #     "features": { ... },   # extracted feature values
    # }
"""

import os
import logging

import numpy as np
import joblib

from feature_extractor import extract_all_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "detector_model.pkl")


# ---------------------------------------------------------------------------
# Bundle loader
# ---------------------------------------------------------------------------

def load_bundle(model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Load the model bundle from *model_path*.

    Returns a dict with keys: model, scaler, imputer, threshold,
    feature_names, include_gemini.

    Raises ``FileNotFoundError`` if the .pkl doesn't exist.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model bundle not found at '{model_path}'. "
            "Run  python train_model.py  first."
        )

    bundle = joblib.load(model_path)
    logger.info("Loaded model bundle from %s  (threshold=%.4f)",
                model_path, bundle["threshold"])
    return bundle


def is_model_available(model_path: str = DEFAULT_MODEL_PATH) -> bool:
    """Check whether the trained model file exists on disk."""
    return os.path.isfile(model_path)


# ---------------------------------------------------------------------------
# False-positive probability
# ---------------------------------------------------------------------------

def compute_false_positive_probability(prob_ai: float, bundle: dict) -> float:
    """Estimate the probability that a *human* text would score ≥ *prob_ai*.

    Uses the sorted array of P(AI) scores for all human samples seen during
    training (``bundle["human_probs_sorted"]``). Returns a value in [0, 1]
    representing the fraction of human texts that reached this confidence
    level — i.e. the empirical false-positive rate at that threshold.

    If the calibration data is missing (old model), returns ``-1``.
    """
    human_probs = bundle.get("human_probs_sorted")
    if human_probs is None or len(human_probs) == 0:
        return -1.0

    # Number of human samples scoring >= prob_ai
    idx = np.searchsorted(human_probs, prob_ai, side="left")
    n_above = len(human_probs) - idx
    return round(float(n_above / len(human_probs)), 6)


def compute_margin_of_error(fp_prob: float, bundle: dict, confidence_level: float = 0.95) -> float:
    """Compute the 95% confidence interval margin of error for the classification threshold.

    Uses standard error calculation: z * sqrt(p * (1 - p) / N)
    """
    human_probs = bundle.get("human_probs_sorted")
    n_samples = len(human_probs) if human_probs is not None and len(human_probs) > 0 else 500
    p = max(min(fp_prob, 0.999), 0.001) if fp_prob >= 0 else 0.05
    z = 1.96  # 95% confidence
    moe = z * np.sqrt((p * (1.0 - p)) / n_samples)
    return round(float(moe * 100), 2)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(text: str, model_path: str = DEFAULT_MODEL_PATH, bundle: dict | None = None, custom_threshold: float = 0.15) -> dict:
    """Classify *text* as human (0) or AI-generated (1).

    Returns
    -------
    dict
        label                  : int — 0 (human) or 1 (AI)
        probability            : float — raw P(AI) from the model
        calibrated_probability : float — Isotonic Regression calibrated P(AI)
        threshold              : float — decision boundary used
        verdict                : str — human-readable verdict
        features               : dict — raw feature values extracted from the text
        false_positive_probability: float — probability human text scores >= prob_ai
        margin_of_error        : float — statistical margin of error (%) for threshold
    """
    if bundle is None:
        bundle = load_bundle(model_path)

    model = bundle["model"]
    scaler = bundle["scaler"]
    imputer = bundle["imputer"]
    # Allow overriding the bundle's dynamically tuned threshold with a strict custom threshold
    threshold = custom_threshold if custom_threshold is not None else bundle["threshold"]
    feature_names = bundle["feature_names"]
    include_gemini = bundle.get("include_gemini", True)

    # --- Extract features ---
    raw_features = extract_all_features(text, include_gemini=True)

    # Build feature vector in the correct order
    feature_vector = np.array(
        [[raw_features.get(f) for f in feature_names]]
    )

    # Impute + scale
    feature_vector = imputer.transform(feature_vector)
    feature_vector = scaler.transform(feature_vector)

    # --- Predict ---
    prob_ai = float(model.predict_proba(feature_vector)[0, 1])

    # --- Isotonic Regression Calibration ---
    iso_cal = bundle.get("isotonic_calibrator")
    if iso_cal is not None:
        try:
            calibrated_prob = float(iso_cal.transform([prob_ai])[0])
        except Exception:
            calibrated_prob = prob_ai
    else:
        calibrated_prob = prob_ai

    label = int(prob_ai >= threshold)

    # False-positive probability
    fp_prob = compute_false_positive_probability(prob_ai, bundle)
    moe = compute_margin_of_error(fp_prob if fp_prob >= 0 else 0.05, bundle)

    # Human-readable verdict
    if prob_ai >= 0.80:
        verdict = "Very likely AI-Generated"
    elif prob_ai >= 0.50:
        verdict = "Likely AI-Generated"
    elif prob_ai >= threshold:
        verdict = "Possibly AI-Generated"
    else:
        verdict = "Human Written"

    return {
        "label": label,
        "probability": round(prob_ai, 4),
        "calibrated_probability": round(calibrated_prob, 4),
        "threshold": threshold,
        "verdict": verdict,
        "features": raw_features,
        "false_positive_probability": fp_prob if fp_prob >= 0 else None,
        "margin_of_error": moe,
    }


def predict_sentences(text: str, model_path: str = DEFAULT_MODEL_PATH, bundle: dict | None = None) -> list[list[dict]]:
    """Split *text* into paragraphs and sentences, then run localized classifier
    inference on each individual sentence.

    Returns a list of paragraphs, where each paragraph is a list of sentence dicts:
    [
        [
            {"text": "Sentence 1 text...", "score": 0.85, "prob_pct": 85.0, "label": 1},
            ...
        ], ...
    ]
    """
    from nltk.tokenize import sent_tokenize

    if bundle is None:
        bundle = load_bundle(model_path)

    model = bundle["model"]
    scaler = bundle["scaler"]
    imputer = bundle["imputer"]
    feature_names = bundle["feature_names"]
    threshold = 0.15  # Strict threshold — matches predict() aggressive default

    paragraphs_raw = text.split("\n")
    paragraphs_result = []

    for para in paragraphs_raw:
        para_trimmed = para.strip()
        if not para_trimmed:
            paragraphs_result.append([])
            continue

        sentences = sent_tokenize(para_trimmed)
        if not sentences:
            sentences = [para_trimmed]

        sentence_dicts = []
        for sent in sentences:
            sent_str = sent.strip()
            if not sent_str:
                continue

            # Fast structural feature extraction (no Gemini API calls for individual sentences)
            raw_feats = extract_all_features(sent_str, include_gemini=False)
            feat_vector = np.array([[raw_feats.get(f) for f in feature_names]])

            feat_imputed = imputer.transform(feat_vector)
            feat_scaled = scaler.transform(feat_imputed)

            prob_ai = float(model.predict_proba(feat_scaled)[0, 1])

            sentence_dicts.append({
                "text": sent_str,
                "score": round(prob_ai, 4),
                "prob_pct": round(prob_ai * 100, 1),
                "label": int(prob_ai >= threshold)
            })

        paragraphs_result.append(sentence_dicts)

    return paragraphs_result


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_texts = [
        (
            "The implementation of sustainable energy solutions requires "
            "a comprehensive understanding of both technological capabilities "
            "and economic constraints. Furthermore, the integration of renewable "
            "resources into existing infrastructure necessitates careful planning."
        ),
        (
            "So yeah I was just walking to the store right and then my buddy "
            "calls me up outta nowhere like hey man you gotta come see this. "
            "I'm like dude what are you talking about lol."
        ),
    ]

    for i, txt in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {txt[:80]}...")
        try:
            result = predict(txt)
            print(f"  Verdict    : {result['verdict']}")
            print(f"  P(AI)      : {result['probability']}")
            print(f"  Label      : {result['label']}")
            print(f"  Threshold  : {result['threshold']}")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
