# =========================
# Chest X-Ray Classifier (MobileNetV2)
# - Full training runs to completion (no EarlyStopping)
# - TensorBoard logging enabled
# - Faster pipeline (cache/prefetch), mixed precision on GPU
# - 2-stage training: head -> fine-tune
# =========================

import os
import zipfile
import warnings
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# ==== REPRODUCIBILITY (best effort) ====
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# ==== GPU CONFIG (optional) ====
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Mixed precision can speed up training on modern GPUs
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled.")
    except Exception as e:
        print("GPU config warning:", e)

# ==== UNZIP DATA ====
zip_path = r'D:\chest_x_ray.zip.zip'  # Change if needed
extract_path = r'D:\chest_x_ray'

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# ==== PATHS ====
path = os.path.join(extract_path, "chest_xray")
assert os.path.exists(path), f"Dataset folder not found at: {path}"

# ==== DATASETS ====
BATCH_SIZE = 32
IMG_SIZE = (256, 256)
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(ds, shuffle=False):
    if shuffle:
        ds = ds.shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

Train = keras.utils.image_dataset_from_directory(
    os.path.join(path, 'train'),
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED
)
Train = preprocess(Train, shuffle=True)

Validation = keras.utils.image_dataset_from_directory(
    os.path.join(path, 'val'),
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)
Validation = preprocess(Validation, shuffle=False)

Test = keras.utils.image_dataset_from_directory(
    os.path.join(path, 'test'),
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)
Test = preprocess(Test, shuffle=False)

class_names = None
try:
    # Newer TF keeps class names on the dataset
    class_names = Train.class_names
except Exception:
    pass

# ==== CLASS WEIGHTS ====
# Build label list by iterating once through the (shuffled) train ds
labels_flat = []
for _, y in Train:
    labels_flat.extend(np.argmax(y.numpy(), axis=1))
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_flat),
    y=labels_flat
)
class_weights = dict(enumerate(class_weights_arr))
print("Class weights:", class_weights)

# ==== DATA AUGMENTATION ====
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="augmentation")

# ==== MODEL ====
base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Stage 1: freeze

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    # Important: output dtype float32 to avoid dtype mismatch with mixed precision
    Dense(2, activation='softmax', dtype='float32')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==== TENSORBOARD ====
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
print(f"TensorBoard logs -> {log_dir}")
# Launch from terminal:  tensorboard --logdir=logs/

# ==== CALLBACKS (no EarlyStopping; all epochs run) ====
callbacks = [
    ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1, min_lr=1e-6),
    tensorboard_cb
]

# ==== STAGE 1: Train classifier head ====
EPOCHS_STAGE1 = 10
print("Stage 1: Training classifier head (frozen base model)...")
history1 = model.fit(
    Train,
    validation_data=Validation,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ==== STAGE 2: Fine-tune ====
# Unfreeze top layers of base_model for fine-tuning
base_model.trainable = True
FREEZE_UP_TO = 100  # keep lower-level features frozen
for layer in base_model.layers[:FREEZE_UP_TO]:
    layer.trainable = False

# Use a smaller LR for fine-tuning
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS_STAGE2 = 30
print("Stage 2: Fine-tuning the model...")
history2 = model.fit(
    Train,
    validation_data=Validation,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ==== COMBINE HISTORIES ====
history = {}
for k in history1.history.keys():
    history[k] = history1.history[k] + history2.history[k]

# ==== PLOTS ====
acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy', marker='o')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='o')
plt.legend(loc='lower right')
plt.grid(True, linestyle="--", alpha=0.6)
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss', marker='o')
plt.plot(epochs_range, val_loss, label='Validation Loss', marker='o')
plt.legend(loc='upper right')
plt.grid(True, linestyle="--", alpha=0.6)
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

# ==== EVALUATION ====
print("Evaluating on test set...")
test_loss, test_acc = model.evaluate(Test, verbose=1)
print(f"\nTest Accuracy: {test_acc:.4f}")

# ==== CONFUSION MATRIX & REPORT ====
y_true, y_pred = [], []
for images, labels in Test:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
try:
    class_names = Test.class_names  # prefer from dataset, if available
except Exception:
    if class_names is None:
        # fallback names
        class_names = [str(i) for i in sorted(set(y_true))]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ==== SAVE FINAL MODEL (in addition to best checkpoint) ====
model.save("final_model.keras")
print("Saved final model to final_model.keras and best checkpoint to best_model.keras")

print(f"\nTo inspect training with TensorBoard, run:\n  tensorboard --logdir=logs/\nThen open the shown URL in your browser.")
