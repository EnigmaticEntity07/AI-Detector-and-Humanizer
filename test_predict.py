from feature_extractor import get_gemini_predictability
import logging
logging.basicConfig(level=logging.DEBUG)

text = "This is a test text."
res = get_gemini_predictability(text)
print("Result:", res)
