"""
feature_extractor.py
====================
Extracts structural and LLM-based features from text for the AI detector
classifier.  All functions are stateless and can be called independently.

Features extracted:
  - avg_sentence_length   : mean word count per sentence
  - vocab_richness        : unique words / total words  (type-token ratio)
  - stopword_freq         : proportion of words that are English stop words
  - sentence_count        : number of sentences
  - avg_word_length       : mean character count per word
  - punctuation_ratio     : punctuation characters / total characters
  - gemini_predictability : 0-100 score from Gemini (optional, None on failure)
"""

import re
import string
import json
import logging

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# ---------------------------------------------------------------------------
# NLTK data bootstrap (safe to call multiple times)
# ---------------------------------------------------------------------------
for _resource in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{_resource}" if "punkt" in _resource else f"corpora/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

_STOP_WORDS = set(stopwords.words("english"))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------

def extract_structural_features(text: str) -> dict:
    """Return a dict of hand-crafted structural features for *text*.

    All features are numeric floats so they can be fed directly into a
    scikit-learn pipeline.
    """
    # --- Sentence-level ---
    sentences = sent_tokenize(text)
    sentence_count = len(sentences) if sentences else 1

    words = word_tokenize(text.lower())
    word_count = len(words) if words else 1

    words_alpha = [w for w in words if w.isalpha()]
    word_count_alpha = len(words_alpha) if words_alpha else 1

    # Average sentence length (words per sentence)
    words_per_sentence = [len(word_tokenize(s)) for s in sentences]
    avg_sentence_length = (
        sum(words_per_sentence) / len(words_per_sentence)
        if words_per_sentence
        else 0.0
    )

    # --- Vocabulary richness (type-token ratio) ---
    unique_words = set(words_alpha)
    vocab_richness = len(unique_words) / word_count_alpha

    # --- Stop-word frequency ---
    stopword_count = sum(1 for w in words_alpha if w in _STOP_WORDS)
    stopword_freq = stopword_count / word_count_alpha

    # --- Average word length ---
    avg_word_length = (
        sum(len(w) for w in words_alpha) / word_count_alpha
        if words_alpha
        else 0.0
    )

    # --- Punctuation ratio ---
    total_chars = len(text) if text else 1
    punct_chars = sum(1 for ch in text if ch in string.punctuation)
    punctuation_ratio = punct_chars / total_chars

    return {
        "avg_sentence_length": float(avg_sentence_length),
        "vocab_richness": float(vocab_richness),
        "stopword_freq": float(stopword_freq),
        "sentence_count": float(sentence_count),
        "avg_word_length": float(avg_word_length),
        "punctuation_ratio": float(punctuation_ratio),
    }


# ---------------------------------------------------------------------------
# Gemini predictability score
# ---------------------------------------------------------------------------

def get_gemini_predictability(text: str, _call_gemini=None) -> float | None:
    """Ask Gemini to rate the text's *predictability* on a 0-100 scale.

    Returns a float in [0, 100] or ``None`` if the call fails for any reason
    (rate-limit, network error, unparseable response, etc.).

    Parameters
    ----------
    text : str
        The text to evaluate.
    _call_gemini : callable, optional
        An override for the Gemini caller (useful for testing).  When ``None``
        the function imports ``call_gemini`` from ``app.py``.
    """
    # Truncate very long texts to avoid token-limit issues
    snippet = text[:3000]

    prompt = (
        "You are a linguistic analyst. Rate the following text's "
        "**predictability** — how likely each next word or phrase is given "
        "the preceding context.  A highly predictable text (common phrasing, "
        "formulaic structure) scores close to 100.  A surprising, creative, "
        "or varied text scores close to 0.\n\n"
        "Return ONLY a JSON object: {\"predictability\": <int 0-100>}\n\n"
        f"Text:\n{snippet}"
    )

    try:
        if _call_gemini is None:
            from app import call_gemini  # lazy import to avoid circular deps
        else:
            call_gemini = _call_gemini

        response = call_gemini(prompt)
        result_text = response.text.strip()

        # Strip markdown fences if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]

        data = json.loads(result_text.strip())
        score = int(data.get("predictability", -1))
        if 0 <= score <= 100:
            return float(score)
        return None
    except Exception as exc:
        logger.debug("Gemini predictability call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Combined feature extraction
# ---------------------------------------------------------------------------

def extract_all_features(text: str, include_gemini: bool = True,
                         _call_gemini=None) -> dict:
    """Return **all** features (structural + Gemini) as a flat dict.

    Parameters
    ----------
    text : str
        Input text.
    include_gemini : bool
        If ``False``, skip the Gemini call entirely (faster for bulk
        structural-only extraction).
    _call_gemini : callable, optional
        Override for the Gemini caller.
    """
    features = extract_structural_features(text)
    if include_gemini:
        features["gemini_predictability"] = get_gemini_predictability(
            text, _call_gemini=_call_gemini
        )
    else:
        features["gemini_predictability"] = None
    return features
