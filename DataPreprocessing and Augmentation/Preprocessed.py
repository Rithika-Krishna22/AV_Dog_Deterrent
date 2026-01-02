# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")  # ultralytics small segmentation model
def preprocess_image(in_path, out_path, output_size=(224, 224)):
    try:
        img = cv2.imread(in_path)
        if img is None:
            return False

        # Predict segmentation
        results = model.predict(img, verbose=False)

        # Check if any mask exists
        if not results[0].masks:
            return False

        # Get mask polygons
        mask_polygons = results[0].masks.xy
        h, w = img.shape[:2]
        binary_mask = np.zeros((h, w), dtype=np.uint8)

        # Convert polygons to binary mask
        for poly in mask_polygons:
            poly = np.array(poly, np.int32)
            cv2.fillPoly(binary_mask, [poly], 1)

        # Blur background
        blurred = cv2.GaussianBlur(img, (21, 21), 0)
        dog_only = np.where(binary_mask[..., None] == 1, img, blurred)

        # Crop bounding box
        ys, xs = np.where(binary_mask == 1)
        if len(xs) == 0 or len(ys) == 0:
            return False
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        roi = dog_only[y_min:y_max, x_min:x_max]

        # Resize to fixed size
        final_img = cv2.resize(roi, output_size)
        cv2.imwrite(out_path, final_img)
        return True

    except Exception as e:
        print("Error:", e)
        return False
# Input folders
input_folders = {
    "Angry": "/content/drive/MyDrive/Datasets/Angry dogs",
    "NonAngry": "/content/drive/MyDrive/Datasets/Non Angry dogs"
}

# Output base folder
output_base = "/content/drive/MyDrive/Datasets/processed"

# Create folders
for key in input_folders:
    os.makedirs(os.path.join(output_base, key), exist_ok=True)


for label, folder in input_folders.items():
    print(f"Processing {label} dogs...")
    out_folder = os.path.join(output_base, label)

    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            in_path = os.path.join(folder, filename)
            out_path = os.path.join(out_folder, filename)
            success = preprocess_image(in_path, out_path, output_size=(224, 224))
            if success:
                print("✅ Processed:", filename)
            else:
                print("⚠️ Skipped:", filename)
