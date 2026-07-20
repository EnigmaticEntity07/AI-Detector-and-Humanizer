"""
train_model.py
==============
End-to-end training pipeline for the AI detector classifier.

Usage
-----
    python train_model.py                           # defaults (2000 samples)
    python train_model.py --sample-size 500         # quick test run
    python train_model.py --csv my_data.csv         # custom dataset
    python train_model.py --no-gemini               # skip Gemini API calls
    python train_model.py --output models/model.pkl # custom output path

The script:
  1. Loads the CSV dataset (must have 'text' and 'label' columns).
  2. Draws a stratified sample of configurable size.
  3. Extracts structural features (+ optional Gemini predictability).
  4. Trains a Logistic Regression classifier with class_weight='balanced'.
  5. Sweeps decision thresholds to maximise **recall** on the AI class (label 1)
     while keeping precision above a configurable floor.
  6. Saves the model bundle (model, scaler, imputer, threshold, feature names)
     to a .pkl file.
"""

import argparse
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

from feature_extractor import extract_all_features

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CSV = "combined_dataset.csv.gz"
DEFAULT_SAMPLE = 2000
DEFAULT_OUTPUT = "detector_model.pkl"
DEFAULT_PRECISION_FLOOR = 0.30   # minimum acceptable precision when tuning
DEFAULT_TEST_SIZE = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature extraction (batch)
# ---------------------------------------------------------------------------

def extract_features_batch(texts: pd.Series, include_gemini: bool = True) -> pd.DataFrame:
    """Extract features for a series of texts, returning a DataFrame."""
    records = []
    for text in tqdm(texts, desc="Extracting features", unit="doc"):
        feats = extract_all_features(
            str(text),
            include_gemini=include_gemini,
        )
        records.append(feats)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
    beta: float = 2.0,
) -> float:
    """Find the threshold that maximises F-beta (default β=2, recall-heavy)
    for label-1 (AI) while keeping precision ≥ *precision_floor*.

    β=2 means recall is weighted **4×** more than precision, so the
    threshold will be biased toward flagging AI aggressively.

    Falls back to 0.35 if no threshold satisfies the constraints.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    best_threshold = 0.35  # aggressive fallback
    best_fbeta = -1.0

    for p, r, t in zip(precisions, recalls, thresholds):
        if p < precision_floor:
            continue
        # F-beta score: (1 + β²) * (p * r) / (β² * p + r)
        denom = (beta ** 2) * p + r
        if denom == 0:
            continue
        fbeta = (1 + beta ** 2) * (p * r) / denom
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold = t

    return float(best_threshold)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train(
    csv_path: str = DEFAULT_CSV,
    sample_size: int = DEFAULT_SAMPLE,
    include_gemini: bool = True,
    output_path: str = DEFAULT_OUTPUT,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
):
    """Run the full training pipeline and save the model bundle."""

    # ---- 1. Load data ----
    logger.info("Loading dataset from %s ...", csv_path)
    df = pd.read_csv(csv_path)

    if "text" not in df.columns or "label" not in df.columns:
        logger.error("CSV must contain 'text' and 'label' columns.")
        sys.exit(1)

    df = df[["text", "label"]].dropna()
    logger.info("Loaded %d rows  (label distribution: %s)",
                len(df), df["label"].value_counts().to_dict())

    # ---- 2. Stratified sample ----
    if sample_size and sample_size < len(df):
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(
                n=min(len(g), sample_size // 2),
                random_state=RANDOM_STATE,
            )
        ).reset_index(drop=True)
        logger.info("Sampled %d rows  (label distribution: %s)",
                     len(df), df["label"].value_counts().to_dict())

    # ---- 3. Extract features ----
    logger.info("Extracting features (gemini=%s) ...", include_gemini)
    X_df = extract_features_batch(df["text"], include_gemini=include_gemini)
    y = df["label"].values

    feature_names = list(X_df.columns)
    logger.info("Features: %s", feature_names)

    # ---- 4. Impute + Scale ----
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X_df.values)
    X_scaled = scaler.fit_transform(X_imputed)

    # ---- 5. Train/test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=DEFAULT_TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # ---- 6. Train Random Forest ----
    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ---- 7. Default threshold report ----
    y_prob_test = model.predict_proba(X_test)[:, 1]

    logger.info("\n=== Classification Report (default threshold = 0.50) ===")
    y_pred_default = (y_prob_test >= 0.50).astype(int)
    logger.info("\n%s", classification_report(y_test, y_pred_default,
                                              target_names=["Human", "AI"]))

    # ---- 8. Tune threshold for high recall ----
    optimal_threshold = find_best_threshold(y_test, y_prob_test, precision_floor)
    logger.info("Optimal threshold (precision floor=%.2f): %.4f",
                precision_floor, optimal_threshold)

    y_pred_tuned = (y_prob_test >= optimal_threshold).astype(int)
    logger.info("\n=== Classification Report (tuned threshold = %.4f) ===",
                optimal_threshold)
    report = classification_report(y_test, y_pred_tuned,
                                   target_names=["Human", "AI"])
    logger.info("\n%s", report)

    cm = confusion_matrix(y_test, y_pred_tuned)
    logger.info("Confusion Matrix:\n%s", cm)

    # ---- 9. Build false-positive calibration data ----
    # For every human sample in the *full* dataset, record P(AI).
    # At inference we can then answer: "What fraction of known-human texts
    # scored ≥ this confidence?" — i.e., the false-positive probability.
    human_mask = (y == 0)
    y_prob_all = model.predict_proba(X_scaled)[:, 1]
    human_probs_sorted = np.sort(y_prob_all[human_mask])  # ascending
    logger.info(
        "False-positive calibration: %d human samples, "
        "max P(AI) among humans = %.4f",
        len(human_probs_sorted),
        human_probs_sorted[-1] if len(human_probs_sorted) else 0.0,
    )

    # ---- 10. Save model bundle ----
    bundle = {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "threshold": optimal_threshold,
        "feature_names": feature_names,
        "include_gemini": include_gemini,
        "human_probs_sorted": human_probs_sorted,
    }
    joblib.dump(bundle, output_path)
    logger.info("Model bundle saved to  %s", output_path)

    return bundle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train the AI detector classifier."
    )
    parser.add_argument(
        "--csv", default=DEFAULT_CSV,
        help=f"Path to the dataset CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE,
        help=f"Number of rows to sample for training (default: {DEFAULT_SAMPLE})",
    )
    parser.add_argument(
        "--no-gemini", action="store_true",
        help="Skip Gemini API calls (train on structural features only)",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output path for the .pkl model bundle (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--precision-floor", type=float, default=DEFAULT_PRECISION_FLOOR,
        help=f"Minimum precision when tuning the threshold (default: {DEFAULT_PRECISION_FLOOR})",
    )

    args = parser.parse_args()

    train(
        csv_path=args.csv,
        sample_size=args.sample_size,
        include_gemini=not args.no_gemini,
        output_path=args.output,
        precision_floor=args.precision_floor,
    )


if __name__ == "__main__":
    main()
