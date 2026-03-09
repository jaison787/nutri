import os
import numpy as np
from PIL import Image


try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    
    # Load classes from file or use default
    CLASSES_PATH = os.path.join(os.path.dirname(__file__), "classes.txt")
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r") as f:
            FOOD_CLASSES = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(FOOD_CLASSES)} custom categories from classes.txt")
    else:
        # Fallback to demo classes if no train has occurred
        FOOD_CLASSES = ["apple_pie", "burger", "pizza", "salad", "sushi"] 
        print("Using default food categories.")

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "food_model.keras")
    
    print("Initializing Food Detection CNN (Transfer Learning Mode)...")
    
    # --- ACADEMICALLY CORRECT ARCHITECTURE ---
    # 1. Start with MobileNetV2 base (weights from ImageNet)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # 2. Add custom layers for Food-101
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x) 
    predictions = tf.keras.layers.Dense(len(FOOD_CLASSES), activation='softmax')(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    # 3. Load fine-tuned weights if they exist
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading fine-tuned weights from {MODEL_PATH}...")
            # We try to load weights onto our architecture
            model.load_weights(MODEL_PATH)
            print("Successfully loaded food-specific weights.")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}. Using base model.")
    else:
        print("Warning: Fine-tuned food_model.keras not found. Running with untrained top layers.")
        print("Tip: Run the training script to improve accuracy.")

    CNN_AVAILABLE = True
except ImportError:
    print("TensorFlow not installed. CNN will be skipped.")
    CNN_AVAILABLE = False

def predict_food(pil_image):
    if not CNN_AVAILABLE:
        return None, 0.0
        
    try:
        # Convert to RGB and resize to 224x224
        img = pil_image.convert('RGB').resize((224, 224))
        
        # Convert to numpy array and preprocess
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Make prediction
        preds = model.predict(x, verbose=0)
        
        # Get the top class
        class_idx = np.argmax(preds[0])
        confidence = float(preds[0][class_idx])
        class_name = FOOD_CLASSES[class_idx].replace('_', ' ')
        
        return class_name, confidence
    except Exception as e:
        print(f"CNN Error: {e}")
        return None, 0.0

if __name__ == "__main__":
    print(f"Detected {len(FOOD_CLASSES)} food categories.")
    # Example usage code remains similarly functional
