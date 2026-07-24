import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from feature_extractor import parse_gemini_response
from detector import predict, predict_sentences

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def test_gemini_parsing():
    print("--- Test 1: Gemini Response Parsing ---")
    s1 = '{"structural_predictability": 0.92, "trope_presence": 0.75}'
    p1, t1 = parse_gemini_response(s1)
    assert p1 == 0.92 and t1 == 0.75, f"Failed s1: {p1}, {t1}"
    
    s2 = "I am 99% confident that this text is AI generated."
    p2, t2 = parse_gemini_response(s2)
    assert p2 == 0.99, f"Failed s2: {p2}"
    
    s3 = "Analysis: structural_predictability: 85%, trope_presence: 40%"
    p3, t3 = parse_gemini_response(s3)
    assert p3 == 0.85 and t3 == 0.40, f"Failed s3: {p3}, {t3}"
    print("Gemini response parsing passed successfully!")

def test_ai_text_hard_override():
    print("\n--- Test 2: AI Text Scoring & Hard Override ---")
    ai_sample = (
        "The implementation of sustainable energy solutions requires a comprehensive "
        "understanding of both technological capabilities and economic constraints. "
        "Furthermore, the integration of renewable resources into existing infrastructure "
        "necessitates careful planning and strategic policy framework alignment. "
        "It is important to note that the tapestry of modern energy grids demands multifaceted solutions."
    )
    
    res = predict(ai_sample)
    print("AI Sample Verdict:", res["verdict"])
    print("AI Sample Probability:", res["probability"])
    print("Hard Override Triggered:", res["debug_stats"]["override_triggered"])
    print("Override Reasons:", res["debug_stats"]["override_reasons"])
    assert res["probability"] >= 0.85, f"Expected AI probability >= 0.85, got {res['probability']}"
    assert res["debug_stats"]["override_triggered"] == True, "Override should be triggered for AI sample"

def test_sentence_heatmap():
    print("\n--- Test 3: Sentence Heatmap Calibration ---")
    ai_sample = (
        "The implementation of sustainable energy solutions requires a comprehensive "
        "understanding of both technological capabilities and economic constraints. "
        "Furthermore, the integration of renewable resources into existing infrastructure "
        "necessitates careful planning and strategic policy framework alignment."
    )
    res = predict(ai_sample)
    sentences = predict_sentences(ai_sample, doc_probability=res["probability"])
    assert len(sentences) > 0, "Sentences list empty"
    first_sent = sentences[0][0]
    print("First sentence text:", first_sent["text"][:40])
    print("First sentence score:", first_sent["score"], "prob_pct:", first_sent["prob_pct"])

if __name__ == "__main__":
    test_gemini_parsing()
    test_ai_text_hard_override()
    test_sentence_heatmap()
    print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
