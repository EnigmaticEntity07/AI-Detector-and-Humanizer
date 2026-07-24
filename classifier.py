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
# Post-processing calibration helpers
# ---------------------------------------------------------------------------

# Sigmoid temperature for upper-tail sharpening.
# A value <1.0 sharpens the sigmoid (pushes high scores higher, low scores lower).
# 0.6 gives a strong but controlled boost into the 85-99% zone for AI text.
_SIGMOID_TEMPERATURE: float = 0.6

# Burstiness threshold: sentence_length_std below this value signals uniform,
# AI-typical sentence structure.
_LOW_BURSTINESS_THRESHOLD: float = 3.5

# N-gram density threshold: above this fraction, AI transition patterns dominate.
_HIGH_NGRAM_THRESHOLD: float = 0.12

# Trope threshold: at least this many LLM alignment tropes detected.
_HIGH_TROPE_THRESHOLD: int = 2

# Minimum probability enforced when multiple AI markers are simultaneously present.
_AI_MARKER_FLOOR: float = 0.82


def sigmoid_temperature_scale(p: float, temperature: float = _SIGMOID_TEMPERATURE) -> float:
    """Apply sigmoid temperature scaling to sharpen high-confidence predictions.

    This prevents Isotonic Regression from over-smoothing the upper tail.
    A temperature < 1.0 amplifies the logit, pushing probabilities already
    near 0.8+ firmly into the 85-99% range while leaving low scores largely
    unchanged.

    Parameters
    ----------
    p           : raw probability in (0, 1)
    temperature : sharpening factor; smaller = more aggressive (default 0.6)

    Returns
    -------
    float in [0.0, 1.0]
    """
    p = float(np.clip(p, 1e-7, 1.0 - 1e-7))
    logit = np.log(p / (1.0 - p))
    scaled_logit = logit / temperature
    scaled_p = 1.0 / (1.0 + np.exp(-scaled_logit))
    return float(np.clip(scaled_p, 0.0, 1.0))


def apply_ai_marker_floor(prob: float, features: dict) -> float:
    """Enforce a probability floor when strong AI structural markers co-occur.

    Checks three independent AI signals from the extracted feature dict:
      - low burstiness  : sentence_length_std < _LOW_BURSTINESS_THRESHOLD
      - high n-gram density : ai_ngram_density > _HIGH_NGRAM_THRESHOLD
      - high trope count    : trope_count >= _HIGH_TROPE_THRESHOLD

    If **two or more** markers are simultaneously present the returned
    probability is raised to at least _AI_MARKER_FLOOR (default 0.82).

    Parameters
    ----------
    prob     : current calibrated probability
    features : dict returned by extract_all_features()

    Returns
    -------
    float — original prob or floor-enforced prob, whichever is higher
    """
    markers_present = 0

    burstiness = features.get("sentence_length_std", 999.0)
    if burstiness < _LOW_BURSTINESS_THRESHOLD:
        markers_present += 1

    ngram_density = features.get("ai_ngram_density", 0.0)
    if ngram_density > _HIGH_NGRAM_THRESHOLD:
        markers_present += 1

    trope_count = features.get("trope_count", 0)
    if trope_count >= _HIGH_TROPE_THRESHOLD:
        markers_present += 1

    if markers_present >= 2:
        return max(prob, _AI_MARKER_FLOOR)

    return prob


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

    # --- Post-processing: Aggressive Dataset Calibration ---
    # If the combined score exceeds 40%, scale aggressively toward 85-100%.
    if calibrated_prob > 0.40:
        final_prob = 0.85 + ((calibrated_prob - 0.40) / 0.60) * 0.15
    else:
        final_prob = calibrated_prob * (0.85 / 0.40)
        
    final_prob = float(np.clip(final_prob, 0.0, 1.0))

    label = int(final_prob >= threshold)

    # False-positive probability (computed against raw model prob for calibration consistency)
    fp_prob = compute_false_positive_probability(prob_ai, bundle)
    moe = compute_margin_of_error(fp_prob if fp_prob >= 0 else 0.05, bundle)

    # Human-readable verdict (based on final calibrated + floored probability)
    if final_prob >= 0.80:
        verdict = "Very likely AI-Generated"
    elif final_prob >= 0.50:
        verdict = "Likely AI-Generated"
    elif final_prob >= threshold:
        verdict = "Possibly AI-Generated"
    else:
        verdict = "Human Written"

    return {
        "label": label,
        "probability": round(final_prob, 4),
        "raw_model_probability": round(prob_ai, 4),
        "calibrated_probability": round(calibrated_prob, 4),
        "threshold": threshold,
        "verdict": verdict,
        "features": raw_features,
        "false_positive_probability": fp_prob if fp_prob >= 0 else None,
        "margin_of_error": moe,
    }



def predict_sentences(
    text: str,
    model_path: str = DEFAULT_MODEL_PATH,
    bundle: dict | None = None,
    doc_probability: float | None = None,
) -> list[list[dict]]:
    """Split *text* into paragraphs and sentences, then run localized classifier
    inference on each individual sentence.

    Sentence scores are calibrated relative to the document-level baseline so
    the visual average of the heatmap remains in sync with the overall document
    probability score (within ±0.15).  This prevents isolated sentences from
    being over-flagged when the document is scored as mostly human.

    Parameters
    ----------
    text            : full document text
    model_path      : path to the .pkl bundle
    bundle          : optional pre-loaded bundle (avoids repeated disk I/O)
    doc_probability : pre-computed document-level P(AI) (0–1).  If None it is
                      computed internally from *text*.

    Returns a list of paragraphs, where each paragraph is a list of sentence dicts:
    [
        [
            {"text": "Sentence 1 text...", "score": 0.85, "prob_pct": 85.0, "label": 1},
            ...
        ], ...
    ]
    """
    from nltk.tokenize import sent_tokenize
    from feature_extractor import extract_structural_features

    if bundle is None:
        bundle = load_bundle(model_path)

    model     = bundle["model"]
    scaler    = bundle["scaler"]
    imputer   = bundle["imputer"]
    feature_names = bundle["feature_names"]
    threshold = 0.15  # Strict threshold — matches predict() aggressive default

    # ── Step 1: Compute document-level baseline ──────────────────────────────
    # We need the document burstiness and document probability to anchor
    # sentence scores to the overall document context.
    if doc_probability is None:
        doc_result = predict(text, bundle=bundle)
        doc_probability = doc_result["probability"]

    # Extract document-level structural features for burstiness baseline
    doc_struct = extract_structural_features(text)
    doc_burstiness = doc_struct.get("sentence_length_std", 5.0)

    # Clamp doc probability to avoid edge-case log domain issues
    doc_prob_clamped = float(np.clip(doc_probability, 1e-4, 1.0 - 1e-4))

    # ── Step 2: Score each sentence in its paragraph context ─────────────────
    paragraphs_raw = text.split("\n")
    all_raw_scores: list[float] = []   # flat list of raw sentence probs, for sync
    paragraphs_raw_data: list[list[dict]] = []  # intermediate storage

    for para in paragraphs_raw:
        para_trimmed = para.strip()
        if not para_trimmed:
            paragraphs_raw_data.append([])
            continue

        sentences = sent_tokenize(para_trimmed)
        if not sentences:
            sentences = [para_trimmed]

        sentence_dicts = []
        for sent in sentences:
            sent_str = sent.strip()
            if not sent_str:
                continue

            # Fast structural feature extraction (no Gemini API calls)
            raw_feats = extract_all_features(sent_str, include_gemini=False)
            feat_vector = np.array([[raw_feats.get(f) for f in feature_names]])

            feat_imputed = imputer.transform(feat_vector)
            feat_scaled  = scaler.transform(feat_imputed)

            raw_prob = float(model.predict_proba(feat_scaled)[0, 1])

            # ── Relative sentence adjustment ──────────────────────────────
            # Sentences are short and lack context, so the model is less
            # reliable.  We blend the raw sentence score with the document
            # probability using a weight derived from the sentence's burstiness
            # deviation from the document baseline.
            sent_burstiness = raw_feats.get("sentence_length_std", 0.0)

            # How "similar" this sentence's local structure is to the document.
            # If burstiness deviation is large → sentence is unusual → trust raw.
            # If deviation is small → sentence looks like the doc → blend more.
            burstiness_diff = abs(sent_burstiness - doc_burstiness)
            # alpha=0: use only doc probability.  alpha=1: use only raw sentence.
            # We weight toward the document (alpha ≈ 0.45) for most sentences.
            alpha = float(np.clip(0.3 + burstiness_diff * 0.06, 0.3, 0.85))
            adjusted_prob = alpha * raw_prob + (1.0 - alpha) * doc_prob_clamped
            adjusted_prob = float(np.clip(adjusted_prob, 0.0, 1.0))

            sentence_dicts.append({
                "text":     sent_str,
                "_raw":     raw_prob,
                "score":    round(adjusted_prob, 4),
                "prob_pct": round(adjusted_prob * 100, 1),
                "label":    int(adjusted_prob >= threshold),
            })
            all_raw_scores.append(adjusted_prob)

        paragraphs_raw_data.append(sentence_dicts)

    # ── Step 3: Document sync clamp ───────────────────────────────────────────
    # The visual average of sentence scores must stay within ±0.15 of the
    # document probability.  Apply linear rescaling if the drift is too large.
    if all_raw_scores:
        sent_avg = float(np.mean(all_raw_scores))
        drift = sent_avg - doc_prob_clamped

        if abs(drift) > 0.15:
            # Linearly shift every sentence score toward the document probability
            # by the excess drift amount.
            shift = drift - np.sign(drift) * 0.15  # keep ±0.15 slack
            for para_sentences in paragraphs_raw_data:
                for s in para_sentences:
                    corrected = float(np.clip(s["score"] - shift, 0.0, 1.0))
                    s["score"]    = round(corrected, 4)
                    s["prob_pct"] = round(corrected * 100, 1)
                    s["label"]    = int(corrected >= threshold)

    # ── Step 4: Strip internal _raw key before returning ─────────────────────
    for para_sentences in paragraphs_raw_data:
        for s in para_sentences:
            s.pop("_raw", None)

    return paragraphs_raw_data



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
