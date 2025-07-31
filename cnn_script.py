"""
CNN for classifying histopathology images (Cancer vs Normal)

Expected folder layout:
histopathology/
├── cancer/
└── normal/

Run:
1. Python 3.9 ONLY
2. pip install -r requirements.txt
"""

# Version Check
import sys
if sys.version_info[:2] != (3, 9):
    raise EnvironmentError(f" Python 3.9 required. Current: {sys.version_info[0]}.{sys.version_info[1]}")

#  Disable TensorFlow optimizations for consistent behavior
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

#  Import Dependencies
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

#  Dataset path
DATASET_PATH = 'histopathology'
IMG_SIZE = 64  # resize to 64x64

def load_images_from_folder(folder, label):
    images, labels = [], []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
        except Exception as e:
            print(f"[!] Failed to load {path}: {e}")
    return images, labels
    
#  Load and prepare data
cancer_imgs, cancer_labels = load_images_from_folder(os.path.join(DATASET_PATH, 'cancer'), 1)
normal_imgs, normal_labels = load_images_from_folder(os.path.join(DATASET_PATH, 'normal'), 0)

X = np.array(cancer_imgs + normal_imgs)
y = np.array(cancer_labels + normal_labels)

#  Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#  Normalize
X_train = X_train / 255.0
X_val = X_val / 255.0

#  Define CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

#  Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#  Train
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

#  Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
