# ==============================================================================
# 1. SETUP AND IMPORTS
# ==============================================================================
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
import shutil

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ==============================================================================
# 2. LOAD THE MODEL AND VALIDATION DATA
# ==============================================================================
# --- Load Model ---
model_path = '/content/drive/My Drive/dog_aggression_model.h5'
print(f"Loading model from: {model_path}")
try:
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# --- Prepare Data ---
zip_path = "/content/drive/My Drive/Datasets/final_dataset.zip"
local_extract_path = "/content/final_dataset"

if not os.path.isdir(local_extract_path):
    print("\nUnzipping the dataset...")
    !unzip -q "{zip_path}" -d "/content/"
    print("Dataset successfully unzipped.")

IMG_SIZE = 224
BATCH_SIZE = 16

# IMPORTANT: We set shuffle=False to ensure predictions and labels line up correctly.
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    local_extract_path,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE),
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    shuffle=False, # <-- IMPORTANT
    seed=42,
    validation_split=0.2,
    subset='validation',
    color_mode='grayscale'
)

# Convert grayscale to RGB to match model's input
def convert_to_rgb(image, label):
    return tf.image.grayscale_to_rgb(image), label

validation_dataset = validation_dataset.map(convert_to_rgb).prefetch(buffer_size=tf.data.AUTOTUNE)
print("\nValidation dataset prepared successfully.")

# ==============================================================================
# 3. GET PREDICTIONS AND TRUE LABELS
# ==============================================================================
print("\nGenerating predictions on the validation set...")
# Get the model's predicted probabilities
all_predictions = model.predict(validation_dataset)
# Get the true labels
all_labels = np.concatenate([y for x, y in validation_dataset], axis=0)

# ==============================================================================
# 4. GENERATE AND PLOT THE CONFUSION MATRIX
# ==============================================================================
# Convert probabilities to class predictions (0 or 1)
predicted_classes = (all_predictions > 0.5).astype(int)

# Calculate the confusion matrix
cm = confusion_matrix(all_labels, predicted_classes)

# Define class names (TensorFlow assigns them alphabetically)
class_names = ['Aggressive', 'Non-Aggressive']

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ==============================================================================
# 5. GENERATE AND PLOT THE ROC CURVE
# ==============================================================================
# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Dashed line for random chance
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of model using EfficientNetV2B0')
plt.legend(loc="lower right")
plt.show()