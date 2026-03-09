import os
import io
import json
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
from model.cnn_model import predict_food, CNN_AVAILABLE
from nutrition_lookup import get_nutrition

# Load environment variables from absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Load predefined nutrition database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NUTRITION_DB = {}
try:
    db_path = os.path.join(BASE_DIR, "nutrition_db.json")
    if os.path.exists(db_path):
        with open(db_path, "r") as f:
            NUTRITION_DB = json.load(f)
            print(f"Loaded {len(NUTRITION_DB)} items from nutrition_db.json")
    else:
        print(f"Warning: nutrition_db.json not found at {db_path}")
except Exception as e:
    print(f"Warning: Could not load local nutrition database: {e}")

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
    print(f"Warning: GEMINI_API_KEY not found in environment. Checking {os.path.join(BASE_DIR, '.env')}")

genai.configure(api_key=api_key)

app = FastAPI(title="Nutri AI Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FoodItem(BaseModel):
    name: str
    calories: int
    protein: str
    carbs: str
    fat: str

class NutritionResponse(BaseModel):
    foods: List[FoodItem] = []
    source: str = "Unknown"
    confidence: float = 0.0
    error: str = None

@app.get("/")
async def root():
    return {"message": "Welcome to Nutri AI Backend API"}

@app.post("/analyze", response_model=NutritionResponse)
async def analyze_food(image: UploadFile = File(...)):
    # Validate file type
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")

    try:
        # Read image data
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))

        # --- 1. LOCAL CNN MODEL PREDICTION & PREDEFINED DATABASE LOOKUP ---
        cnn_pred, confidence = None, 0.0
        if CNN_AVAILABLE:
            cnn_pred, confidence = predict_food(pil_image)
            print(f"CNN Model predicted: '{cnn_pred}' with confidence {confidence:.2f}", flush=True)
            
            # --- HYBRID AI LOGIC: Only trust CNN if confidence is high (e.g. >= 0.6) ---
            if confidence >= 0.6 and cnn_pred:
                # 1. Check manual JSON database
                if cnn_pred in NUTRITION_DB:
                    print(f"High confidence match found in local JSON database: '{cnn_pred}'", flush=True)
                    data = NUTRITION_DB[cnn_pred]
                    return {
                        "foods": [
                            {
                                "name": cnn_pred.title(),
                                "calories": data["calories"],
                                "protein": data["protein"],
                                "carbs": data["carbs"],
                                "fat": data["fat"]
                            }
                        ],
                        "source": "Local CNN + JSON DB",
                        "confidence": confidence
                    }
                
                # 2. Check CSV database
                print(f"Searching CSV database for high-confidence match: '{cnn_pred}'...")
                csv_data = get_nutrition(cnn_pred)
                if csv_data:
                    print(f"Found '{cnn_pred}' in CSV database: {csv_data['name']}")
                    return {
                        "foods": [
                            {
                                "name": csv_data["name"],
                                "calories": csv_data["calories"],
                                "protein": csv_data["protein"],
                                "carbs": csv_data["carbs"],
                                "fat": csv_data["fat"]
                            }
                        ],
                        "source": "Local CNN + CSV Lookup",
                        "confidence": confidence
                    }
                else:
                    print(f"'{cnn_pred}' not found in CSV database even with high confidence.", flush=True)
        
        # --- 2. AI FALLBACK (GEMINI) ---
        # Triggered if CNN confidence is low (< 0.6) or not found in databases
        reason = "Low Confidence" if (confidence or 0) < 0.6 else "Not in database"
        print(f"Falling back to Gemini AI Vision (Reason: {reason}, Prediction='{cnn_pred if cnn_pred else 'N/A'}', Confidence={confidence:.2f})", flush=True)

        # Try different model names that are confirmed to work in 2026
        available_models = [
            'models/gemini-2.5-flash',
            'models/gemini-2.0-flash',
            'models/gemini-pro-latest',
            'models/gemini-flash-latest'
        ]
        model = None
        
        if not api_key:
             raise HTTPException(status_code=500, detail="Gemini API Key missing.")

        # Construct Prompt
        prompt = """
        Analyze the food items in this image and provide nutritional information for each.
        Return the result ONLY as a valid JSON object with the following structure:
        {
          "foods": [
            {
              "name": "food name",
              "calories": numerical_value,
              "protein": "value with unit (e.g. 10g)",
              "carbs": "value with unit (e.g. 20g)",
              "fat": "value with unit (e.g. 5g)"
            }
          ]
        }
        Ensure the 'calories' field is an integer. If you see multiple items, list them all.
        If you are unsure about the nutrition, provide your best estimate based on standard portions.
        Do not include any other text or markdown formatting markers like ```json.
        """

        for model_name in available_models:
            print(f"Attempting to use Gemini model: {model_name}...", flush=True)
            try:
                model = genai.GenerativeModel(model_name)
                # Test connectivity/existence with a simple prompt
                response = model.generate_content([prompt, pil_image])
                print(f"Successfully used {model_name}", flush=True)
                break
            except Exception as e:
                print(f"Error with {model_name}: {e}", flush=True)
                model = None
                continue
        
        if not model:
            # Check if it was an API key error
            error_msg = "All Gemini model attempts failed."
            if not api_key:
                error_msg += " API Key is missing or invalid."
            else:
                error_msg += " This usually means the API key is invalid, quota is exceeded, or the model is unavailable."
            
            # Instead of just raising, return a helpful error
            return {
                "foods": [],
                "error": error_msg,
                "source": "AI Fallback Failed",
                "confidence": 0.0
            }

        # --- 3. PARSE RESPONSE ---
        try:
            # Clean response text in case it contains markdown code blocks
            res_text = response.text.strip()
            if res_text.startswith("```json"):
                res_text = res_text[7:]
            if res_text.endswith("```"):
                res_text = res_text[:-3]
            res_text = res_text.strip()
            
            nutrition_data = json.loads(res_text)
            return nutrition_data
        except json.JSONDecodeError as je:
            print(f"JSON Parsing Error: {je}")
            print(f"Raw Response: {response.text}")
            raise HTTPException(status_code=500, detail="Failed to parse AI response into JSON format.")

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50, flush=True)
    print("SERVERS STARTING - NUTRI APP BACKEND READY", flush=True)
    print("PORT: 8003", flush=True)
    print("="*50 + "\n", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8003)
