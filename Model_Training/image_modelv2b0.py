# ==============================================================================
# CELL 1: SETUP AND IMPORTS
# ==============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import os
import shutil

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ==============================================================================
# CELL 2: LOAD AND PREPARE DATA FROM THE ZIP FILE
# ==============================================================================
# --- 1. Define Paths ---
zip_path = "/content/drive/My Drive/Datasets/final_dataset.zip"
local_extract_path = "/content/final_dataset"

# --- 2. Unzip the dataset locally ---
print("Unzipping the dataset...")
if os.path.exists(local_extract_path):
    shutil.rmtree(local_extract_path) # Clean up old directory
!unzip -q "{zip_path}" -d "/content/"
print("Dataset successfully unzipped.")

# --- 3. Load the data using Keras utility ---
IMG_SIZE = 224
BATCH_SIZE = 16 # Keep this smaller to conserve memory

train_dataset, validation_dataset = tf.keras.utils.image_dataset_from_directory(
    local_extract_path,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_SIZE, IMG_SIZE),
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='both',
    color_mode='grayscale'
)

# --- 4. Convert grayscale images to RGB for the pre-trained model ---
def convert_to_rgb(image, label):
    return tf.image.grayscale_to_rgb(image), label

train_dataset = train_dataset.map(convert_to_rgb)
validation_dataset = validation_dataset.map(convert_to_rgb)

# --- 5. Configure dataset for performance (REMOVED .cache() TO PREVENT RAM CRASH) ---
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

print("\nDatasets created and configured successfully.")

# ==============================================================================
# CELL 3: BUILD THE MODEL
# ==============================================================================
input_shape = (IMG_SIZE, IMG_SIZE, 3)
base_model = tf.keras.applications.EfficientNetV2B0(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False
inputs = keras.Input(shape=input_shape)
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

print("\nModel built successfully:")
model.summary()

# ==============================================================================
# CELL 4: COMPILE THE MODEL
# ==============================================================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy(name='accuracy')]
)
print("\nModel compiled successfully.")

# ==============================================================================
# CELL 5: TRAIN THE MODEL
# ==============================================================================
early_stopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
EPOCHS = 20

print("\nStarting model training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopper]
)
print("\nTraining finished.")

# ==============================================================================
# CELL 6: EVALUATE THE RESULTS
# ==============================================================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

print("\nEvaluating final model on the validation set...")
val_loss, val_acc = model.evaluate(validation_dataset)
print(f"\nFinal Validation Accuracy: {val_acc*100:.2f}%")

# ==============================================================================
# CELL 7: SAVE THE FINAL MODEL
# ==============================================================================
model_save_path = '/content/drive/My Drive/dog_aggression_model.h5'
model.save(model_save_path)
print(f"\nModel saved successfully to: {model_save_path}")