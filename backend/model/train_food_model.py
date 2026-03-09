import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# This script is the "Brain" of your project. 
# It demonstrates how to perform TRANSFER LEARNING for Food Detection.

def build_model(num_classes):
    print(f"Building Model Architecture for {num_classes} food categories...")
    
    # 1. Base Model: MobileNetV2 (Pre-trained on ImageNet)
    # include_top=False means we remove the original 1000-class classifier
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base (Do not train the ImageNet layers)
    base_model.trainable = False
    
    # 2. Custom Classification Head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2), 
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_demo_data(data_dir):
    """Creates a tiny synthetic dataset so you can demonstrate training immediately."""
    print("Creating tiny demo dataset for training demonstration...")
    # Classes we want to train for
    classes = ['pizza', 'hamburger', 'sushi', 'salad']
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        os.makedirs(cls_path, exist_ok=True)
        # Create 10 fake images per class
        for i in range(10):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            tf.keras.utils.save_img(os.path.join(cls_path, f"demo_{i}.jpg"), img)
    return classes

if __name__ == "__main__":
    # Settings
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "food_data")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "food_model.keras")
    
    # 1. Prepare Data
    if not os.path.exists(DATA_DIR):
        classes = create_demo_data(DATA_DIR)
    else:
        classes = sorted(os.listdir(DATA_DIR))
    
    print(f"Classes found: {classes}")
    
    # 2. Load Dataset using TensorFlow Utilities
    # This is the industry-standard way to load images
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    
    # 3. Build & Train
    model = build_model(len(classes))
    
    print("\nStarting Training Session...")
    print("Academic Note: We are training only the top layers (Transfer Learning).")
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1 # Reduced for demonstration speed
    )
    
    # 4. Save the Result & Classes
    print(f"\nTraining Complete! Saving model to: {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    
    classes_path = os.path.join(BASE_DIR, "classes.txt")
    with open(classes_path, "w") as f:
        f.write("\n".join(classes))
    print(f"Saved class labels to {classes_path}")
    
    print("\n" + "="*50)
    print("SUCCESS: You have now trained your own Food AI model.")
    print("You can copy these 'food_model.keras' weights to use in your app.")
    print("="*50)
