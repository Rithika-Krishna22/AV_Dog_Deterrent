import tensorflow as tf
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
DATA_DIR = "final_dataset"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20

print("Loading dataset...")
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='both',
    seed=42,
    color_mode='grayscale'
)

def convert_to_rgb(image, label):
    return tf.image.grayscale_to_rgb(image), label

train_ds = train_ds.map(convert_to_rgb).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(convert_to_rgb).prefetch(tf.data.AUTOTUNE)

# Extract true labels for evaluation
y_true = np.concatenate([y for x, y in val_ds], axis=0).flatten()

# ==========================================
# 2. MODEL BUILDER FUNCTION (CNNs)
# ==========================================
def build_model(model_name):
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    
    if model_name == 'Custom_CNN':
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
    else:
        if model_name == 'VGG16':
            base_model = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
        elif model_name == 'ResNet50':
            base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
        elif model_name == 'EfficientNetV2B0':
            base_model = tf.keras.applications.EfficientNetV2B0(input_shape=input_shape, include_top=False, weights='imagenet')
            
        base_model.trainable = False 
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==========================================
# 3. TRAINING DEEP LEARNING MODELS
# ==========================================
models_to_test = ['Custom_CNN', 'VGG16', 'ResNet50', 'EfficientNetV2B0']
results = []
trained_models_dict = {} # Save models in memory to use for the SVM later

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
)

for name in models_to_test:
    print(f"\n{'='*50}\nTraining {name}\n{'='*50}")
    model = build_model(name)
    
    start_time = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])
    train_time = time.time() - start_time
    
    trained_models_dict[name] = model # Store for feature extraction
    
    print(f"Evaluating {name}...")
    y_pred_probs = model.predict(val_ds).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = np.mean(y_true == y_pred)
    
    results.append({
        'Model': name,
        'Accuracy (%)': accuracy * 100,
        'F1-Score': f1,
        'Parameters': f"{model.count_params():,}",
        'Training Time (s)': round(train_time, 2)
    })
    model.save(f"{name}_best.h5")

# ==========================================
# 4. TRAINING THE SVM (Hybrid Architecture)
# ==========================================
print(f"\n{'='*50}\nTraining SVM (Using EfficientNet Features)\n{'='*50}")

def extract_features(dataset, feature_extractor):
    features, labels = [], []
    for images, lbls in dataset:
        feat = feature_extractor.predict(images, verbose=0)
        features.append(feat.reshape(feat.shape[0], -1))
        labels.append(lbls.numpy())
    return np.vstack(features), np.concatenate(labels)

# We use EfficientNetV2B0 as the feature extractor since it is the most optimized
eff_net = trained_models_dict['EfficientNetV2B0']
feature_extractor = tf.keras.Model(
    inputs=eff_net.input,
    outputs=eff_net.layers[-3].output # Tap into the model before the final Dense layers
)

print("Extracting features from training set...")
X_train, y_train_svm = extract_features(train_ds, feature_extractor)

print("Extracting features from validation set...")
X_val, y_val_svm = extract_features(val_ds, feature_extractor)

print("Training SVM...")
start_time = time.time()
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', verbose=True)
svm_model.fit(X_train, y_train_svm.ravel())
train_time = time.time() - start_time

print("Evaluating SVM...")
y_pred_svm = svm_model.predict(X_val)
precision, recall, f1, _ = precision_recall_fscore_support(y_val_svm, y_pred_svm, average='binary')
accuracy = np.mean(y_val_svm == y_pred_svm)

results.append({
    'Model': 'Hybrid SVM',
    'Accuracy (%)': accuracy * 100,
    'F1-Score': f1,
    'Parameters': 'N/A (Scikit-Learn)', # SVMs don't use standard NN parameters
    'Training Time (s)': round(train_time, 2)
})

joblib.dump(svm_model, 'Hybrid_SVM_best.joblib')

# ==========================================
# 5. FINAL COMPARISON & VISUALIZATION
# ==========================================
df = pd.DataFrame(results)
print("\n\n" + "="*80)
print("FINAL MODEL COMPARISON (Including SVM)")
print("="*80)
print(df.to_string(index=False))

fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar chart for Accuracy
color = 'tab:blue'
ax1.set_xlabel('Model Architecture', fontweight='bold')
ax1.set_ylabel('Accuracy (%)', color=color, fontweight='bold')
bars = ax1.bar(df['Model'], df['Accuracy (%)'], color=color, alpha=0.7, width=0.4, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0, 100])

# Line chart for Training Time on a second y-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Training Time (seconds)', color=color, fontweight='bold')
line = ax2.plot(df['Model'], df['Training Time (s)'], color=color, marker='o', linewidth=2, markersize=8, label='Time (s)')
ax2.tick_params(axis='y', labelcolor=color)

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=10)

plt.title('Architecture Comparison: Deep Learning vs. Hybrid SVM', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig("model_comparison_with_svm.png", dpi=300)
plt.show()

print("\nEvaluation complete. Comparison chart saved as 'model_comparison_with_svm.png'.")
