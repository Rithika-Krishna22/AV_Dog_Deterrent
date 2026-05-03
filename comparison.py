import tensorflow as tf
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    color_mode='grayscale' # Assuming your dataset is grayscale
)

# Convert grayscale to RGB for pre-trained models
def convert_to_rgb(image, label):
    return tf.image.grayscale_to_rgb(image), label

train_ds = train_ds.map(convert_to_rgb).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(convert_to_rgb).prefetch(tf.data.AUTOTUNE)

# Extract true labels for later evaluation
y_true = np.concatenate([y for x, y in val_ds], axis=0).flatten()

# ==========================================
# 2. MODEL BUILDER FUNCTION
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
        # Load pre-trained base depending on the selected architecture
        if model_name == 'VGG16':
            base_model = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
        elif model_name == 'ResNet50':
            base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
        elif model_name == 'EfficientNetV2B0':
            base_model = tf.keras.applications.EfficientNetV2B0(input_shape=input_shape, include_top=False, weights='imagenet')
            
        base_model.trainable = False # Freeze base weights
        
        # Attach custom head
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
# 3. TRAINING & EVALUATION LOOP
# ==========================================
models_to_test = ['Custom_CNN', 'VGG16', 'ResNet50', 'EfficientNetV2B0']
results = []

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
)

for name in models_to_test:
    print(f"\n{'='*50}\nTraining {name}\n{'='*50}")
    model = build_model(name)
    
    # Train
    start_time = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])
    train_time = time.time() - start_time
    
    # Predict
    print(f"Evaluating {name}...")
    y_pred_probs = model.predict(val_ds).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Calculate Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = np.mean(y_true == y_pred)
    
    # Store Results
    results.append({
        'Model': name,
        'Accuracy (%)': accuracy * 100,
        'F1-Score': f1,
        'Parameters': f"{model.count_params():,}", # Formats with commas for readability
        'Training Time (s)': round(train_time, 2)
    })
    
    # Save the model
    model.save(f"{name}_best.h5")

# ==========================================
# 4. FINAL COMPARISON & VISUALIZATION
# ==========================================
df = pd.DataFrame(results)
print("\n\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)
print(df.to_string(index=False))

# Plotting the results
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

# Add value labels on top of the bars for clarity
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=10)

plt.title('Architecture Comparison: Accuracy vs. Compute Time', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()

print("\nEvaluation complete. Comparison chart saved as 'model_comparison.png'.")
