import os
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
    
    print("Loading CNN Model...")
    # Load the pre-trained Convolutional Neural Network (CNN) model
    model = MobileNetV2(weights='imagenet')
    CNN_AVAILABLE = True
except ImportError:
    print("TensorFlow not installed. CNN will be skipped.")
    CNN_AVAILABLE = False

def predict_food(pil_image):
    if not CNN_AVAILABLE:
        return None, 0.0
        
    try:
        # Resize image to 224x224 as required by the CNN
        img = pil_image.resize((224, 224))
        
        # Convert to numpy array and preprocess
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Make prediction with the CNN
        preds = model.predict(x, verbose=0)
        
        # Decode top prediction
        results = decode_predictions(preds, top=1)[0]
        top_pred = results[0]
        
        # Format: class_id, class_name, confidence
        class_name = top_pred[1].lower().replace('_', ' ')
        confidence = float(top_pred[2])
        
        return class_name, confidence
    except Exception as e:
        print(f"CNN Error: {e}")
        return None, 0.0
