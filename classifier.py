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
    training (``bundle["human_probs_sorted"]``).  Returns a value in [0, 1]
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


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(text: str, model_path: str = DEFAULT_MODEL_PATH, bundle: dict | None = None, custom_threshold: float = 0.35) -> dict:
    """Classify *text* as human (0) or AI-generated (1).

    Returns
    -------
    dict
        label        : int — 0 (human) or 1 (AI)
        probability  : float — P(AI) from the model
        threshold    : float — decision boundary used
        verdict      : str — human-readable verdict
        features     : dict — raw feature values extracted from the text
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
    # Always extract Gemini feature for UI display, regardless of model training flags
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
    label = int(prob_ai >= threshold)

    # False-positive probability (only meaningful when flagged as AI)
    fp_prob = (
        compute_false_positive_probability(prob_ai, bundle)
        if label == 1
        else None
    )

    # Human-readable verdict
    if prob_ai >= 0.80:
        verdict = "Very likely AI-Generated"
    elif prob_ai >= threshold:
        verdict = "Likely AI-Generated"
    elif prob_ai >= 0.40:
        verdict = "Possibly AI-Generated"
    else:
        verdict = "Human Written"

    return {
        "label": label,
        "probability": round(prob_ai, 4),
        "threshold": threshold,
        "verdict": verdict,
        "features": raw_features,
        "false_positive_probability": fp_prob,
    }


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
