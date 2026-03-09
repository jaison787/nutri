import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    response = model.generate_content("Hello, respond with 'KEY_WORKS' if you can see this.")
    print(response.text)
except Exception as e:
    print(f"FAILED: {e}")
