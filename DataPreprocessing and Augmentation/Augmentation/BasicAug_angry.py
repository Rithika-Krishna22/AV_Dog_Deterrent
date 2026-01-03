# ==============================================================================
# STEP 1: SETUP AND INSTALLATION
# ==============================================================================
# Install the Albumentations library for fast and flexible image augmentation
!pip install albumentations tqdm

from google.colab import drive
import os
import cv2
import albumentations as A
from tqdm import tqdm # For a nice progress bar

# Mount your Google Drive to access the files
drive.mount('/content/drive', force_remount=True)

# ==============================================================================
# STEP 2: CONFIGURE PATHS AND AUGMENTATION SETTINGS
# ==============================================================================
# This is both the INPUT and OUTPUT folder
data_dir = "/content/drive/My Drive/Datasets/aggressive gray scale"

# --- How many new versions to create for EACH original image ---
NUM_AUGMENTATIONS_PER_IMAGE = 7

print(f"Augmenting images in: {data_dir}")
print(f"Creating {NUM_AUGMENTATIONS_PER_IMAGE} new versions for each original image.")

# ==============================================================================
# STEP 3: DEFINE THE AUGMENTATION PIPELINE
# ==============================================================================
# This pipeline defines the random changes that will be applied to the images.
# These are chosen to be effective for grayscale images with transparency.
transform = A.Compose([
    # Rotates, scales, and shifts the image. The empty space will be transparent.
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=20, p=0.8,
                       border_mode=cv2.BORDER_CONSTANT, value=[0,0,0,0]),

    # Randomly changes brightness and contrast.
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),

    # Adds a bit of blur to simulate motion or a slightly out-of-focus camera.
    A.MotionBlur(blur_limit=5, p=0.5),

    # Randomly removes small patches from the image (occlusion).
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.5)
])

# ==============================================================================
# STEP 4: MAIN AUGMENTATION LOOP
# ==============================================================================
# Get a list of the ORIGINAL images before we start adding new ones
original_image_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.png')]
print(f"\nFound {len(original_image_files)} original images to augment.")

# Loop through each original image and create augmented versions
for filename in tqdm(original_image_files, desc="Augmenting Images"):
    image_path = os.path.join(data_dir, filename)

    # Read the image, keeping all 4 channels (for transparency)
    # cv2.IMREAD_UNCHANGED is crucial here
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Warning: Could not read {filename}. Skipping.")
        continue

    # Get the base filename without the .png extension
    base_name = os.path.splitext(filename)[0]

    # Create N augmented versions for this image
    for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
        try:
            # Apply the augmentation pipeline
            augmented = transform(image=image)
            augmented_image = augmented['image']

            # Create a new, unique filename for the augmented image
            output_filename = f"{base_name}_aug_{i+1}.png"
            output_path = os.path.join(data_dir, output_filename)

            # Save the new image
            cv2.imwrite(output_path, augmented_image)
        except Exception as e:
            print(f"An error occurred while augmenting {filename}: {e}")


print("\n🎉 Augmentation complete!")
print(f"The folder '{os.path.basename(data_dir)}' now contains the original and augmented images.")