import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

try:
    models = list(genai.list_models())
    print(f"Total models: {len(models)}")
    for m in models:
        print(f"Model: {m.name}")
except Exception as e:
    print(f"Error: {e}")
