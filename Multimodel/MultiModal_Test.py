import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import matplotlib.pyplot as plt
import os
import joblib
import pickle
import librosa
from google.colab import drive
from google.colab import files

# ==============================================================================
# 1. SETUP & LOAD ALL MODELS
# ==============================================================================
print("🔄 Loading All Models...")

if not os.path.exists('/content/drive'): drive.mount('/content/drive')

# --- A. LOAD VISUAL MODEL ---
image_model_path = '/content/drive/My Drive/dog_aggression_model.h5'
if os.path.exists(image_model_path):
    image_model = tf.keras.models.load_model(image_model_path)
    print("✅ Visual Model Loaded")
else:
    print("❌ Visual Model Not Found!")

# --- B. LOAD AUDIO AGGRESSION MODELS ---
cnn_path = "/content/drive/MyDrive/audio_cnn_model.h5"
ensemble_path = "/content/drive/MyDrive/audio_ensemble_classifier.pkl"

if os.path.exists(cnn_path) and os.path.exists(ensemble_path):
    cnn_model = tf.keras.models.load_model(cnn_path)
    ensemble = pickle.load(open(ensemble_path, "rb"))

    # Feature Extractor for Ensemble
    feature_model = tf.keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer("feature_output").output
    )
    print("✅ Audio Aggression Models Loaded")
else:
    print("❌ Audio Models Not Found!")

# --- C. LOAD YAMNET (DOG DETECTOR) ---
print("⏳ Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
class_names = np.loadtxt(class_map_path, dtype=str, delimiter=',', skiprows=1, usecols=2)
print("✅ YAMNet Loaded")

SAMPLE_RATE = 16000
TARGET_SAMPLES = int(SAMPLE_RATE * 5.0)

# ==============================================================================
# 2. VISUAL HELPER FUNCTIONS
# ==============================================================================
def get_gradcam_and_prediction(img_tensor, full_model, base_layer_name):
    # Robust Layer Finding
    layers = full_model.layers
    base_layer_index = None
    for i, layer in enumerate(layers):
        if layer.name == base_layer_name:
            base_layer_index = i
            break

    base_model = layers[base_layer_index]
    head_layers = layers[base_layer_index + 1:]

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_tensor, tf.float32)
        features = base_model(inputs)
        tape.watch(features)
        x = features
        for layer in head_layers: x = layer(x)
        preds = x
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, features)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    features = features[0]
    heatmap = features @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), preds.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = np.uint8(255 * img)
    return tf.keras.preprocessing.image.array_to_img(jet * alpha + img)

# ==============================================================================
# 3. AUDIO HELPER FUNCTIONS
# ==============================================================================
def audio_to_logmel(y, sr):
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
    else:
        y = y[:TARGET_SAMPLES]
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
    return librosa.power_to_db(mel, ref=np.max)

def analyze_audio_pipeline(file_path):
    try:
        # Load Audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # 1. YAMNet Check (Is it a dog?)
        scores, embeddings, spectrogram = yamnet_model(y)
        dog_keywords = ["Dog", "Bark", "Bow-wow", "Woof", "Growling", "Yelp", "Howl"]
        dog_indices = [i for i, name in enumerate(class_names) if any(k.lower() in name.lower() for k in dog_keywords)]

        dog_scores = tf.gather(scores, indices=dog_indices, axis=1)
        dog_mean = float(np.mean(dog_scores))

        # Threshold: If mean score is < 0.10, there is likely no dog sound
        if dog_mean < 0.10:
            return 0.0, f"🔇 Silent / No Dog (Mean: {dog_mean:.2f})"

        # 2. Aggression Check (If dog is present)
        logmel = audio_to_logmel(y, sr)
        inp = np.expand_dims(logmel, axis=(0, -1))

        cnn_prob = float(cnn_model.predict(inp, verbose=0)[0][0])
        embedding = feature_model.predict(inp, verbose=0).reshape(1, -1)
        ensemble_prob = float(ensemble.predict_proba(embedding)[0][1])

        final_audio_prob = (cnn_prob + ensemble_prob) / 2
        return final_audio_prob, f"🔊 Dog Detected (Aggression: {final_audio_prob:.1%})"

    except Exception as e:
        # If librosa fails (e.g., uploaded an image), treat as Silence
        return 0.0, "🔇 Audio Error / Image File"

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
print("\n👇 UPLOAD FILE (Video or Image) 👇")
uploaded = files.upload()

for filename in uploaded.keys():
    try:
        print(f"\n🎬 Processing: {filename}...")

        # --- A. VISUAL ANALYSIS ---
        # Robust Frame Extraction (Works for Video OR Image files)
        cap = cv2.VideoCapture(filename)
        ret, frame = cap.read() # Read first frame
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = tf.image.resize(frame_rgb, [224, 224])
            img_tensor = np.expand_dims(tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img_resized)), axis=0)

            heatmap, raw_vis_preds = get_gradcam_and_prediction(img_tensor, image_model, 'efficientnetv2-b0')
            visual_prob = 1.0 - raw_vis_preds[0][0]
        else:
            visual_prob = 0.0
            print("❌ Could not read visual data.")

        # --- B. AUDIO ANALYSIS ---
        audio_prob, audio_status = analyze_audio_pipeline(filename)

        # --- C. THE SMART FUSION LOGIC ---

        # CASE 1: SILENCE / NO DOG HEARD
        if "Silent" in audio_status or "Error" in audio_status:
            # 🛑 REALITY CHECK: If silent, penalize the visual score significantly.
            # We cut the visual score by 50% to prevent hallucinations.
            # e.g., Bathtub Dog (91%) -> becomes 45.5% (Non-Aggressive)
            final_score = visual_prob * 0.5
            logic_msg = "🔇 Silence Penalty Applied (Score Halved)"

        # CASE 2: DOG HEARD
        else:
            # ⚠️ DANGER CHECK: If either model is screaming "Aggressive" (>85%), trust it.
            if audio_prob > 0.85 or visual_prob > 0.85:
                final_score = max(audio_prob, visual_prob)
                logic_msg = "⚠️ High Danger Trigger (Max Score)"
            else:
                # Standard Fusion
                final_score = (visual_prob * 0.5) + (audio_prob * 0.5)
                logic_msg = "⚖️ Standard Weighted Average"

        final_label = "⚠️ AGGRESSIVE" if final_score > 0.50 else "✅ NON-AGGRESSIVE"

        # --- D. DISPLAY ---
        print("\n" + "="*45)
        print(f"📊 FINAL REPORT FOR: {filename}")
        print("="*45)
        print(f"👁️ Visual Score:     {visual_prob:.1%}")
        print(f"🔊 Audio Score:      {audio_prob:.1%} ({audio_status})")
        print("-" * 25)
        print(f"🧠 Logic Used:       {logic_msg}")
        print(f"⚡ FINAL SCORE:      {final_score:.1%}")
        print(f"🏆 PREDICTION:       {final_label}")
        print("="*45)

        if ret:
            result_img = overlay_heatmap(cv2.resize(frame_rgb, (224, 224))/255.0, heatmap)
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1); plt.imshow(frame_rgb); plt.title("Input Visual"); plt.axis('off')
            plt.subplot(1, 2, 2); plt.imshow(result_img); plt.title("AI Focus"); plt.axis('off')
            plt.show()

    except Exception as e:
        print(f"❌ Error: {e}")