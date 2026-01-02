# ============================================
# 1. INSTALL & IMPORT LIBRARIES
# ============================================
# Install Gradio and Ultralytics (for YOLOv8)
!pip install gradio ultralytics opencv-python-headless -q

import gradio as gr
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from ultralytics import YOLO

print("Libraries installed and imported successfully.")

# ============================================
# 2. LOAD MODELS
# ============================================
# --- A. Load Your Trained Aggression Classifier ---
print("⏳ Loading aggression classifier...")
# TYPE THE CORRECT PATH TO YOUR MODEL HERE:
model_path = '/content/drive/My Drive/dog_aggression_model.h5'
try:
    classifier_model = keras.models.load_model(model_path)
    print("✅ Classifier loaded successfully!")
except IOError:
    print(f"❌ ERROR: Could not find model at {model_path}. Please check the path.")

# --- B. Load YOLOv8 Object Detector ---
print("⏳ Loading YOLOv8 dog detector...")
# 'yolov8n.pt' is the Nano model: fast and small.
# It will download automatically on first run.
yolo_model = YOLO('yolov8n.pt')
print("✅ YOLO detector ready!")

# ============================================
# 3. CORE PREDICTION LOGIC (The Smart Part)
# ============================================
def crop_and_classify(image):
    """
    1. Uses YOLO to find a dog.
    2. Crops the image to just the dog.
    3. Sends the cropped image to the classifier.
    """
    # --- Step 1: Detect with YOLO ---
    # Run YOLO on the image. It returns a list of results.
    results = yolo_model(image, verbose=False)

    # Best dog found so far
    best_dog_box = None
    highest_confidence = 0.0

    # Iterate through detections to find the best "dog" (COCO class index 16)
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # Class 16 is 'dog' in the COCO dataset that YOLO is trained on
        if class_id == 16:
            if confidence > highest_confidence:
                highest_confidence = confidence
                # Get coordinates [x1, y1, x2, y2] and convert to integers
                best_dog_box = box.xyxy[0].cpu().numpy().astype(int)

    # --- Step 2: Crop (or use full image if no dog found) ---
    if best_dog_box is not None:
        # Found a dog! Crop the image.
        x1, y1, x2, y2 = best_dog_box
        # Ensure coordinates are within image bounds
        h, w, _ = image.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        cropped_image = image[y1:y2, x1:x2]
        status_msg = f"✅ Dog detected (Conf: {highest_confidence:.2f}). Cropping image."
        # (Optional) Draw box on original image for visualization later if needed
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        # No dog found. Use the whole image as a fallback.
        cropped_image = image
        status_msg = "⚠️ No dog detected by YOLO. Analyzing entire image."

    # --- Step 3: Prepare for Classifier ---
    # Resize the (cropped) image to 224x224
    img_resized = tf.image.resize(cropped_image, [224, 224])
    # Add batch dimension [1, 224, 224, 3]
    img_array = np.expand_dims(img_resized, axis=0)

    # --- Step 4: Classify Aggression ---
    prediction = classifier_model.predict(img_array, verbose=0)
    score = prediction[0][0]

    # Recall: score closer to 0 is Aggressive, 1 is Non-Aggressive
    non_aggressive_prob = score
    aggressive_prob = 1 - score

    # Return labels and the cropped image so the user sees what the AI saw
    labels = {"Non-Aggressive": float(non_aggressive_prob), "Aggressive": float(aggressive_prob)}
    return labels, cropped_image, status_msg

# ============================================
# 4. LAUNCH GRADIO INTERFACE
# ============================================
# We add output image and text boxes to show the cropping process
interface = gr.Interface(
    fn=crop_and_classify,
    inputs=gr.Image(label="1. Upload an Image containing a Dog"),
    outputs=[
        gr.Label(num_top_classes=2, label="3. Final Prediction"),
        gr.Image(label="2. What the AI actually analyzed (Cropped Area)"),
        gr.Textbox(label="Process Status")
    ],
    title="🐶 Smart Dog Aggression Detector (YOLO + Classifier)",
    description="First, YOLO detects and crops the dog. Then, the classifier analyzes only the cropped area for aggression."
)

print("Starting Gradio...")
# Set debug=True to see errors in Colab output if something goes wrong
interface.launch(share=True, debug=True)