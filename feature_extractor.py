"""
feature_extractor.py
====================
Extracts structural and LLM-based features from text for the AI detector
classifier. All functions are stateless and can be called independently.

Features extracted:
  - avg_sentence_length       : mean word count per sentence
  - sentence_length_std       : standard deviation of words per sentence (burstiness)
  - vocab_richness            : unique words / total words (type-token ratio)
  - stopword_freq             : proportion of words that are English stop words
  - sentence_count            : number of sentences
  - avg_word_length           : mean character count per word
  - punctuation_ratio         : punctuation characters / total characters
  - flesch_kincaid_grade      : readability score via textstat
  - paragraph_symmetry        : standard deviation of word counts across paragraphs
  - trope_count               : frequency of common LLM alignment tropes
  - gemini_predictability     : 0-1 score from Gemini (optional, None on failure)
  - gemini_trope_presence     : 0-1 score from Gemini (optional, None on failure)
"""

import re
import string
import json
import logging

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import textstat

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

# Hardcoded list of common LLM tropes
LLM_TROPES = [
    "delve", "tapestry", "it is important to note", "in conclusion",
    "not merely a story of", "the next great leap", "moreover", "furthermore",
    "testament to", "rich tapestry", "treasure trove", "it's worth noting",
    "shed light on", "navigating the", "landscape", "crucial", "embark on",
    "realm", "beacon", "pivotal", "multifaceted"
]

# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------

def extract_structural_features(text: str) -> dict:
    """Return a dict of hand-crafted structural features for *text*."""
    # --- Sentence-level ---
    sentences = sent_tokenize(text)
    sentence_count = len(sentences) if sentences else 1

    words = word_tokenize(text.lower())
    word_count = len(words) if words else 1

    words_alpha = [w for w in words if w.isalpha()]
    word_count_alpha = len(words_alpha) if words_alpha else 1

    # Sentence lengths
    words_per_sentence = [len(word_tokenize(s)) for s in sentences]
    avg_sentence_length = np.mean(words_per_sentence) if words_per_sentence else 0.0
    sentence_length_std = np.std(words_per_sentence) if words_per_sentence else 0.0

    # Paragraph Symmetry (variance in paragraph length)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    words_per_paragraph = [len(word_tokenize(p)) for p in paragraphs]
    paragraph_symmetry = np.std(words_per_paragraph) if len(words_per_paragraph) > 1 else 0.0

    # Type-Token Ratio
    unique_words = set(words_alpha)
    vocab_richness = len(unique_words) / word_count_alpha

    # Stop-word frequency
    stopword_count = sum(1 for w in words_alpha if w in _STOP_WORDS)
    stopword_freq = stopword_count / word_count_alpha

    # Average word length
    avg_word_length = (
        sum(len(w) for w in words_alpha) / word_count_alpha
        if words_alpha
        else 0.0
    )

    # Punctuation ratio
    total_chars = len(text) if text else 1
    punct_chars = sum(1 for ch in text if ch in string.punctuation)
    punctuation_ratio = punct_chars / total_chars

    # Flesch Kincaid Grade
    flesch_kincaid = textstat.flesch_kincaid_grade(text)

    # Trope Count
    text_lower = text.lower()
    trope_count = sum(text_lower.count(trope) for trope in LLM_TROPES)

    return {
        "avg_sentence_length": float(avg_sentence_length),
        "sentence_length_std": float(sentence_length_std),
        "vocab_richness": float(vocab_richness),
        "stopword_freq": float(stopword_freq),
        "sentence_count": float(sentence_count),
        "avg_word_length": float(avg_word_length),
        "punctuation_ratio": float(punctuation_ratio),
        "flesch_kincaid_grade": float(flesch_kincaid),
        "paragraph_symmetry": float(paragraph_symmetry),
        "trope_count": float(trope_count)
    }


# ---------------------------------------------------------------------------
# Gemini predictability score
# ---------------------------------------------------------------------------

def get_gemini_features(text: str, _call_gemini=None) -> dict:
    """Ask Gemini to rate the text for LLM tropes and structural predictability.

    Returns a dict with `gemini_predictability` and `gemini_trope_presence`
    (both scaled 0.0 - 1.0). Returns None values if the call fails.
    """
    snippet = text[:3000]

    prompt = (
        "You are a linguistic analyst. Rate the following text on two dimensions:\n"
        "1. **structural_predictability**: how likely each next word or phrase is given "
        "the preceding context (0.0 = highly creative/varied, 1.0 = highly formulaic).\n"
        "2. **trope_presence**: the presence of common LLM alignment tropes like 'delve', 'tapestry', 'it is important to note' "
        "(0.0 = no tropes, 1.0 = very heavy usage).\n\n"
        "Return ONLY a JSON object exactly like this: {\"structural_predictability\": 0.85, \"trope_presence\": 0.40}\n\n"
        f"Text:\n{snippet}"
    )

    result = {"gemini_predictability": None, "gemini_trope_presence": None}
    
    try:
        if _call_gemini is None:
            from app import call_gemini  # lazy import
        else:
            call_gemini = _call_gemini

        response = call_gemini(prompt)
        result_text = response.text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:]
        elif result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]

        data = json.loads(result_text.strip())
        
        pred = data.get("structural_predictability")
        trope = data.get("trope_presence")
        
        if pred is not None and 0 <= float(pred) <= 1.0:
            result["gemini_predictability"] = float(pred)
        if trope is not None and 0 <= float(trope) <= 1.0:
            result["gemini_trope_presence"] = float(trope)
            
    except Exception as exc:
        logger.debug("Gemini predictability call failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Combined feature extraction
# ---------------------------------------------------------------------------

def extract_all_features(text: str, include_gemini: bool = True,
                         _call_gemini=None) -> dict:
    """Return **all** features (structural + Gemini) as a flat dict."""
    features = extract_structural_features(text)
    
    if include_gemini:
        gemini_feats = get_gemini_features(text, _call_gemini=_call_gemini)
        features.update(gemini_feats)
    else:
        features["gemini_predictability"] = None
        features["gemini_trope_presence"] = None
        
    return features
