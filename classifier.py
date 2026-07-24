"""
classifier.py
=============
Inference module for the trained AI detector classifier.
Re-exports primary detection functions from detector.py.
"""

from detector import (
    predict,
    predict_sentences,
    load_bundle,
    is_model_available,
    compute_false_positive_probability,
    compute_margin_of_error,
    sigmoid_temperature_scale,
    apply_ai_marker_floor,
    DEFAULT_MODEL_PATH,
)

__all__ = [
    "predict",
    "predict_sentences",
    "load_bundle",
    "is_model_available",
    "compute_false_positive_probability",
    "compute_margin_of_error",
    "sigmoid_temperature_scale",
    "apply_ai_marker_floor",
    "DEFAULT_MODEL_PATH",
]
