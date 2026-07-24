"""
detector.py
===========
Core detection pipeline module for AI text classification.

Features:
  - Gemini output parsing with regex fallback.
  - Label mapping verification (dynamic lookup for class 1 = AI).
  - Robust combined scoring logic with Hard Override rule (snapping to 85-100% on high AI signals).
  - Comprehensive logging.info() statements at each step.
  - Sentence-level heatmap calculation maintaining document calibration.
"""

import os
import logging
import numpy as np
import joblib

from feature_extractor import extract_all_features, extract_structural_features

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_SIGMOID_TEMPERATURE: float = 0.6
_LOW_BURSTINESS_THRESHOLD: float = 3.5
_HIGH_NGRAM_THRESHOLD: float = 0.12
_HIGH_TROPE_THRESHOLD: int = 2
_AI_MARKER_FLOOR: float = 0.85

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "detector_model.pkl")


def sigmoid_temperature_scale(p: float, temperature: float = _SIGMOID_TEMPERATURE) -> float:
    """Apply sigmoid temperature scaling to sharpen predictions."""
    p = float(np.clip(p, 1e-7, 1.0 - 1e-7))
    logit = np.log(p / (1.0 - p))
    scaled_logit = logit / temperature
    scaled_p = 1.0 / (1.0 + np.exp(-scaled_logit))
    return float(np.clip(scaled_p, 0.0, 1.0))


def apply_ai_marker_floor(prob: float, features: dict) -> float:
    """Enforce probability floor when multiple AI markers co-occur."""
    markers_present = 0
    if features.get("sentence_length_std", 999.0) < _LOW_BURSTINESS_THRESHOLD:
        markers_present += 1
    if features.get("ai_ngram_density", 0.0) > _HIGH_NGRAM_THRESHOLD:
        markers_present += 1
    if features.get("trope_count", 0) >= _HIGH_TROPE_THRESHOLD:
        markers_present += 1
    if markers_present >= 2:
        return max(prob, _AI_MARKER_FLOOR)
    return prob


def load_bundle(model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Load the trained model bundle from *model_path*."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model bundle not found at '{model_path}'. Run python train_model.py first."
        )
    bundle = joblib.load(model_path)
    logger.info("Loaded detector model bundle from %s (threshold=%.4f)", model_path, bundle.get("threshold", 0.15))
    return bundle


def is_model_available(model_path: str = DEFAULT_MODEL_PATH) -> bool:
    """Check whether the trained model file exists on disk."""
    return os.path.isfile(model_path)


def compute_false_positive_probability(prob_ai: float, bundle: dict) -> float:
    """Estimate empirical false-positive probability for human text."""
    human_probs = bundle.get("human_probs_sorted")
    if human_probs is None or len(human_probs) == 0:
        return -1.0
    idx = np.searchsorted(human_probs, prob_ai, side="left")
    n_above = len(human_probs) - idx
    return round(float(n_above / len(human_probs)), 6)


def compute_margin_of_error(fp_prob: float, bundle: dict, confidence_level: float = 0.95) -> float:
    """Compute standard margin of error (%) for threshold."""
    human_probs = bundle.get("human_probs_sorted")
    n_samples = len(human_probs) if human_probs is not None and len(human_probs) > 0 else 500
    p = max(min(fp_prob, 0.999), 0.001) if fp_prob >= 0 else 0.05
    z = 1.96
    moe = z * np.sqrt((p * (1.0 - p)) / n_samples)
    return round(float(moe * 100), 2)


def predict(
    text: str,
    model_path: str = DEFAULT_MODEL_PATH,
    bundle: dict | None = None,
    custom_threshold: float = 0.15,
) -> dict:
    """Classify text as human (0) or AI-generated (1) with pipeline logging and debug stats."""
    if bundle is None:
        bundle = load_bundle(model_path)

    model = bundle["model"]
    scaler = bundle["scaler"]
    imputer = bundle["imputer"]
    threshold = custom_threshold if custom_threshold is not None else bundle.get("threshold", 0.15)
    feature_names = bundle["feature_names"]

    # 1. Pipeline Logging: Raw input
    word_count = len(text.split())
    char_count = len(text)
    snippet = text[:150].replace("\n", " ")
    logger.info("PIPELINE [Raw Input]: %d chars, %d words | Snippet: '%s...'", char_count, word_count, snippet)

    # 2. Extract features
    raw_features = extract_all_features(text, include_gemini=True)
    raw_gemini_str = raw_features.get("raw_gemini_response") or "None / Failed"
    logger.info("PIPELINE [Gemini Output String]: '%s'", raw_gemini_str)

    burstiness = raw_features.get("sentence_length_std", 0.0)
    ngram_density = raw_features.get("ai_ngram_density", 0.0)
    trope_count = raw_features.get("trope_count", 0.0)
    gemini_pred = raw_features.get("gemini_predictability")
    gemini_trope = raw_features.get("gemini_trope_presence")

    logger.info(
        "PIPELINE [Extracted Raw Features]: burstiness(std)=%.3f, ai_ngram_density=%.3f, trope_count=%d, gemini_predictability=%s, gemini_trope_presence=%s",
        burstiness,
        ngram_density,
        int(trope_count),
        str(gemini_pred),
        str(gemini_trope),
    )

    # Build feature vector
    feature_vector = np.array([[raw_features.get(f) for f in feature_names]])
    feature_vector = imputer.transform(feature_vector)
    feature_vector = scaler.transform(feature_vector)

    # 3. Label Inversion Check & Model Prediction
    # Dynamically find the column index corresponding to class 1 (AI)
    classes = getattr(model, "classes_", np.array([0, 1]))
    if 1 in classes:
        ai_idx = int(np.where(classes == 1)[0][0])
    else:
        ai_idx = 1

    probs = model.predict_proba(feature_vector)[0]
    prob_ai = float(probs[ai_idx])
    logger.info("PIPELINE [Unscaled Classifier Score]: P(AI)=%.4f (class mapping: index %d -> label 1)", prob_ai, ai_idx)

    # Isotonic calibration
    iso_cal = bundle.get("isotonic_calibrator")
    if iso_cal is not None:
        try:
            calibrated_prob = float(iso_cal.transform([prob_ai])[0])
        except Exception:
            calibrated_prob = prob_ai
    else:
        calibrated_prob = prob_ai

    # 4. Combined Scoring & Hard Override Rule
    override_triggered = False
    override_reasons = []

    sentence_count = raw_features.get("sentence_count", 0.0)

    if gemini_pred is not None and gemini_pred > 0.60:
        override_triggered = True
        override_reasons.append(f"Gemini raw predictability ({gemini_pred:.2f}) > 60%")

    if sentence_count >= 4 and burstiness < _LOW_BURSTINESS_THRESHOLD:
        override_triggered = True
        override_reasons.append(f"Low burstiness ({burstiness:.2f} < {_LOW_BURSTINESS_THRESHOLD}) across {int(sentence_count)} sentences")

    if ngram_density > _HIGH_NGRAM_THRESHOLD:
        override_triggered = True
        override_reasons.append(f"High AI n-gram density ({ngram_density:.3f} > {_HIGH_NGRAM_THRESHOLD})")

    if trope_count >= _HIGH_TROPE_THRESHOLD:
        override_triggered = True
        override_reasons.append(f"High trope count ({int(trope_count)} >= {_HIGH_TROPE_THRESHOLD})")

    if override_triggered:
        # Snap aggressively to 85%-100% range
        signal_strength = max(
            gemini_pred if gemini_pred is not None else 0.0,
            prob_ai,
            (ngram_density / 0.20) if ngram_density > 0 else 0.0,
            0.50,
        )
        override_score = 0.85 + 0.15 * float(np.clip((signal_strength - 0.50) / 0.50, 0.0, 1.0))
        final_prob = max(calibrated_prob, override_score)
        logger.info("PIPELINE [Hard Override Triggered]: Reasons: %s | Final score snapped to %.4f", ", ".join(override_reasons), final_prob)
    else:
        if calibrated_prob > 0.40:
            final_prob = 0.85 + ((calibrated_prob - 0.40) / 0.60) * 0.15
        else:
            final_prob = calibrated_prob * (0.85 / 0.40)
        logger.info("PIPELINE [Standard Scaling]: Calibrated score %.4f -> Final score %.4f", calibrated_prob, final_prob)

    final_prob = float(np.clip(final_prob, 0.0, 1.0))
    label = int(final_prob >= threshold)

    fp_prob = compute_false_positive_probability(prob_ai, bundle)
    moe = compute_margin_of_error(fp_prob if fp_prob >= 0 else 0.05, bundle)

    if final_prob >= 0.80:
        verdict = "Very likely AI-Generated"
    elif final_prob >= 0.50:
        verdict = "Likely AI-Generated"
    elif final_prob >= threshold:
        verdict = "Possibly AI-Generated"
    else:
        verdict = "Human Written"

    debug_stats = {
        "text_chars": char_count,
        "text_words": word_count,
        "raw_gemini_response": raw_gemini_str,
        "burstiness_std": round(burstiness, 3),
        "ai_ngram_density": round(ngram_density, 4),
        "trope_count": int(trope_count),
        "gemini_predictability": gemini_pred,
        "gemini_trope_presence": gemini_trope,
        "unscaled_model_probability": round(prob_ai, 4),
        "calibrated_probability": round(calibrated_prob, 4),
        "override_triggered": override_triggered,
        "override_reasons": override_reasons,
        "final_probability": round(final_prob, 4),
        "label": label,
    }

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
        "debug_stats": debug_stats,
    }


def predict_sentences(
    text: str,
    model_path: str = DEFAULT_MODEL_PATH,
    bundle: dict | None = None,
    doc_probability: float | None = None,
) -> list[list[dict]]:
    """Sentence-level heatmap predictions anchored to document probability baseline."""
    from nltk.tokenize import sent_tokenize

    if bundle is None:
        bundle = load_bundle(model_path)

    model = bundle["model"]
    scaler = bundle["scaler"]
    imputer = bundle["imputer"]
    feature_names = bundle["feature_names"]
    threshold = 0.15

    classes = getattr(model, "classes_", np.array([0, 1]))
    ai_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else 1

    if doc_probability is None:
        doc_result = predict(text, bundle=bundle)
        doc_probability = doc_result["probability"]

    doc_struct = extract_structural_features(text)
    doc_burstiness = doc_struct.get("sentence_length_std", 5.0)
    doc_prob_clamped = float(np.clip(doc_probability, 1e-4, 1.0 - 1e-4))

    paragraphs_raw = text.split("\n")
    all_raw_scores: list[float] = []
    paragraphs_raw_data: list[list[dict]] = []

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

            raw_feats = extract_all_features(sent_str, include_gemini=False)
            feat_vector = np.array([[raw_feats.get(f) for f in feature_names]])
            feat_imputed = imputer.transform(feat_vector)
            feat_scaled = scaler.transform(feat_imputed)

            raw_prob = float(model.predict_proba(feat_scaled)[0, ai_idx])

            sent_burstiness = raw_feats.get("sentence_length_std", 0.0)
            burstiness_diff = abs(sent_burstiness - doc_burstiness)
            alpha = float(np.clip(0.3 + burstiness_diff * 0.06, 0.3, 0.85))
            adjusted_prob = alpha * raw_prob + (1.0 - alpha) * doc_prob_clamped
            adjusted_prob = float(np.clip(adjusted_prob, 0.0, 1.0))

            sentence_dicts.append({
                "text": sent_str,
                "_raw": raw_prob,
                "score": round(adjusted_prob, 4),
                "prob_pct": round(adjusted_prob * 100, 1),
                "label": int(adjusted_prob >= threshold),
            })
            all_raw_scores.append(adjusted_prob)

        paragraphs_raw_data.append(sentence_dicts)

    if all_raw_scores:
        sent_avg = float(np.mean(all_raw_scores))
        drift = sent_avg - doc_prob_clamped
        if abs(drift) > 0.15:
            shift = drift - np.sign(drift) * 0.15
            for para_sentences in paragraphs_raw_data:
                for s in para_sentences:
                    corrected = float(np.clip(s["score"] - shift, 0.0, 1.0))
                    s["score"] = round(corrected, 4)
                    s["prob_pct"] = round(corrected * 100, 1)
                    s["label"] = int(corrected >= threshold)

    for para_sentences in paragraphs_raw_data:
        for s in para_sentences:
            s.pop("_raw", None)

    return paragraphs_raw_data


if __name__ == "__main__":
    sample_ai_text = (
        "The implementation of sustainable energy solutions requires a comprehensive "
        "understanding of both technological capabilities and economic constraints. "
        "Furthermore, the integration of renewable resources into existing infrastructure "
        "necessitates careful planning and strategic policy framework alignment."
    )
    res = predict(sample_ai_text)
    print("Direct Execution Test Output:")
    print("Verdict:", res["verdict"])
    print("Probability:", res["probability"])
    print("Debug Stats:", res["debug_stats"])
