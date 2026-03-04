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

# Load environment variables
load_dotenv()

# Load predefined nutrition database
NUTRITION_DB = {}
try:
    with open("nutrition_db.json", "r") as f:
        NUTRITION_DB = json.load(f)
except Exception as e:
    print(f"Warning: Could not load local nutrition database: {e}")

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
    print("Warning: GEMINI_API_KEY is not set correctly in .env file.")

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
    foods: List[FoodItem]

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
        if CNN_AVAILABLE:
            cnn_pred, confidence = predict_food(pil_image)
            print(f"CNN Model predicted: '{cnn_pred}' with confidence {confidence:.2f}")
            
            # If CNN is confident and we have the food in our predefined database
            if confidence > 0.35 and cnn_pred and cnn_pred in NUTRITION_DB:
                print(f"Found '{cnn_pred}' in local database. Serving predefined nutrition values.")
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
                    ]
                }
        
        print("CNN confidence too low or item not in local DB. Falling back to Gemini AI...")

        # --- 2. AI FALLBACK (GEMINI) ---
        # Try different model names that are confirmed to work
        available_models = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash']
        model = None
        for model_name in available_models:
            try:
                model = genai.GenerativeModel(model_name)
                # Check if it exists by doing a dry run or just proceeding
                break
            except Exception:
                continue
        
        if not model:
            raise HTTPException(status_code=500, detail="No suitable Gemini model found.")

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

        # Generate content
        response = model.generate_content([prompt, pil_image])
        
        # Parse response text to JSON
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
