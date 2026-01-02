# ==============================================================================
# STEP 1: SETUP AND INSTALLATION
# ==============================================================================
# Install the Ultralytics library for YOLOv8 and tqdm for a progress bar
!pip install ultralytics tqdm

from google.colab import drive
import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm # For a nice progress bar

# Mount your Google Drive to access the files
drive.mount('/content/drive', force_remount=True)

# ==============================================================================
# STEP 2: CONFIGURE YOUR FOLDER PATHS
# ==============================================================================
# Path to the folder with the original non-angry dog images
input_dir = "/content/drive/My Drive/Datasets/Not angry dogs"

# Path for the NEW folder where the processed non-angry images will be saved
output_dir = "/content/drive/My Drive/Datasets/non_aggressive_gray_scale"

# Create the output directory if it doesn't already exist
os.makedirs(output_dir, exist_ok=True)
print(f"Reading original images from: {input_dir}")
print(f"Saving processed images to NEW folder: {output_dir}")

# ==============================================================================
# STEP 3: THE IMAGE PROCESSING FUNCTION
# ==============================================================================
FIXED_RESOLUTION = (224, 224)
DOG_CLASS_ID = 16 # COCO class ID for 'dog'

def process_image_for_transparent_dog(image_path, output_path, model):
    """
    Detects and segments a dog, creates a transparent background around it,
    converts the dog to grayscale, crops tightly, and resizes.
    """
    try:
        img = cv2.imread(image_path)
        if img is None: return

        results = model(img, verbose=False)

        dog_detections = []
        for result in results:
            if result.masks is not None:
                for i, box in enumerate(result.boxes):
                    if int(box.cls) == DOG_CLASS_ID:
                        dog_detections.append({
                            'mask': result.masks.data[i].cpu().numpy(),
                            'conf': box.conf.item()
                        })

        if not dog_detections: return

        best_detection = max(dog_detections, key=lambda x: x['conf'])
        mask = best_detection['mask']

        h, w = img.shape[:2]

        # --- THIS IS THE CORRECTED LINE ---
        scaled_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        binary_mask = (scaled_mask > 0.5).astype(np.uint8) * 255

        y_coords, x_coords = np.where(binary_mask == 255)

        if y_coords.size == 0 or x_coords.size == 0: return

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        transparent_img = np.zeros((h, w, 4), dtype=np.uint8)
        transparent_img[:, :, 3] = binary_mask
        transparent_img[:, :, :3] = img

        transparent_img[:, :, 0] = transparent_img[:, :, 0] * (binary_mask // 255)
        transparent_img[:, :, 1] = transparent_img[:, :, 1] * (binary_mask // 255)
        transparent_img[:, :, 2] = transparent_img[:, :, 2] * (binary_mask // 255)

        cropped_dog_alpha = transparent_img[y_min:y_max, x_min:x_max]
        if cropped_dog_alpha.size == 0: return

        cropped_dog_bgr = cropped_dog_alpha[:, :, :3]
        cropped_dog_alpha_mask = cropped_dog_alpha[:, :, 3]
        gray_dog_bgr = cv2.cvtColor(cropped_dog_bgr, cv2.COLOR_BGR2GRAY)
        gray_dog_transparent = cv2.merge([gray_dog_bgr, gray_dog_bgr, gray_dog_bgr, cropped_dog_alpha_mask])

        resized_dog = cv2.resize(gray_dog_transparent, FIXED_RESOLUTION, interpolation=cv2.INTER_AREA)

        cv2.imwrite(output_path, resized_dog)

    except Exception as e:
        print(f"An error occurred while processing {os.path.basename(image_path)}: {e}")

# ==============================================================================
# STEP 4: MAIN EXECUTION BLOCK
# ==============================================================================
print("\nLoading YOLOv8 segmentation model...")
model = YOLO("yolov8m-seg.pt")
print("Model loaded successfully.")

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"\nFound {len(image_files)} images to process. Starting...")

for filename in tqdm(image_files, desc="Processing Non-Aggressive Images"):
    input_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}.png"
    output_path = os.path.join(output_dir, output_filename)

    process_image_for_transparent_dog(input_path, output_path, model)

print("\n🎉 Processing complete!")
print(f"All new processed images have been saved to: {output_dir}")