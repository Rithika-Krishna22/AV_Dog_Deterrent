import tensorflow as tf
from tensorflow import keras
import os
import shutil

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ==============================================================================
# 1. LOAD YOUR SAVED MODEL
# ==============================================================================
model_path = '/content/drive/My Drive/dog_aggression_model.h5'
print(f"Loading model from: {model_path}")

try:
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    model.summary()
except Exception as e:
    print(f"Error loading model: {e}")
    # Stop execution if the model can't be loaded
    raise

# ==============================================================================
# 2. PREPARE THE VALIDATION DATASET
# ==============================================================================
# --- Define Paths and Unzip ---
zip_path = "/content/drive/My Drive/Datasets/final_dataset.zip"
local_extract_path = "/content/final_dataset"

if not os.path.isdir(local_extract_path):
    print("\nUnzipping the dataset for evaluation...")
    !unzip -q "{zip_path}" -d "/content/"
    print("Dataset successfully unzipped.")

# --- Load ONLY the validation subset ---
IMG_SIZE = 224
BATCH_SIZE = 16

# We use 'subset="validation"' to only load the 20% validation split
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    local_extract_path,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE),
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    shuffle=False, # No need to shuffle for evaluation
    seed=42,
    validation_split=0.2,
    subset='validation', # <-- IMPORTANT
    color_mode='grayscale'
)

# --- Convert grayscale images to RGB (must match training) ---
def convert_to_rgb(image, label):
    return tf.image.grayscale_to_rgb(image), label

validation_dataset = validation_dataset.map(convert_to_rgb)

# --- Configure for performance ---
AUTOTUNE = tf.data.AUTOTUNE
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

print("\nValidation dataset prepared successfully.")

# ==============================================================================
# 3. EVALUATE THE MODEL AND PRINT ACCURACY
# ==============================================================================
print("\nEvaluating model performance on the validation set...")
loss, accuracy = model.evaluate(validation_dataset)

print("-" * 30)
print(f"Final Model Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Final Model Validation Loss: {loss:.4f}")