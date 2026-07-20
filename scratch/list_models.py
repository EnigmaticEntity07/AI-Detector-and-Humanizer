import google.generativeai as genai
import toml

try:
    config = toml.load(open('.streamlit/secrets.toml'))
    api_key = config.get('GEMINI_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        print("Available models:")
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                print(f" - {m.name}")
    else:
        print("API key not found in secrets.toml")
except Exception as e:
    print(f"Error: {e}")
