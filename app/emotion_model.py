"""Facial emotion CNN -- a faithful port of "Model 3" from Selena's completed MIT
Professional Education capstone (Facial_Emotion_Detection.ipynb).

This is NOT a fresh architecture invented for this app. It is the exact 5-conv-block
+ 2-dense-block network that, in the original notebook, beat a plain baseline CNN,
a batchnorm CNN, and three ImageNet transfer-learning heads (VGG16, ResNet101,
EfficientNetV2B2) on the same 4-class dataset (happy / sad / neutral / surprise,
48x48 grayscale):

    Model 1 (baseline 3-block CNN)              66.4% test accuracy
    Model 2 (4-block CNN, batchnorm+LeakyReLU)   71.1% test accuracy
    VGG16 (transfer learning, frozen)            50.0% test accuracy
    ResNet101 (transfer learning, frozen)        25.0% test accuracy (chance level)
    EfficientNetV2B2 (transfer learning, frozen) 25.0% test accuracy (chance level)
    Model 3 (this architecture)                  77.3% test accuracy  <- winner

Per-class test performance (128-image held-out test set):

                precision  recall  f1-score  support
    happy            0.81    0.91      0.85       32
    sad              0.70    0.59      0.64       32
    neutral          0.64    0.72      0.68       32
    surprise         0.97    0.88      0.92       32

The transfer-learning models underperforming a from-scratch CNN is a real, documented
finding in the notebook, not a bug: ImageNet backbones expect ~224x224 inputs, and
resizing them down to this dataset's native 48x48 resolution destroyed most of the
pretrained features they rely on.

IMPORTANT: this module defines and can run the architecture, but the *trained weights*
(model3.weights.h5) live wherever the original Colab session saved them -- they were
not available in this session. Without them, `EmotionClassifier.predict()` runs on
whatever weights are loaded (random-initialized, if none are provided), and its output
is not a meaningful prediction. See scripts/export_weights_from_colab.md for how to
get the real trained weights into `models/model3.weights.h5`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    LeakyReLU,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential

IMG_SIZE = 48
CLASS_LABELS = ["happy", "sad", "neutral", "surprise"]


def build_model3(img_size: int = IMG_SIZE, no_of_classes: int = 4) -> Sequential:
    """Reconstructs Model 3 exactly as defined in the capstone notebook (cell 91)."""
    model = Sequential(name="model3_facial_emotion_cnn")

    # 1st CNN block
    model.add(Conv2D(64, kernel_size=2, padding="same", input_shape=(img_size, img_size, 1), activation="relu"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    # 2nd CNN block
    model.add(Conv2D(128, kernel_size=2, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    # 3rd CNN block
    model.add(Conv2D(512, kernel_size=2, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    # 4th CNN block
    model.add(Conv2D(512, kernel_size=2, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    # 5th CNN block
    model.add(Conv2D(128, kernel_size=2, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Flatten())

    # First fully connected block
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    # Second fully connected block
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(no_of_classes, activation="softmax"))
    return model


class EmotionClassifier:
    def __init__(self, weights_path: str | Path | None = None) -> None:
        self.model = build_model3()
        self.weights_loaded = False
        if weights_path is not None and Path(weights_path).exists():
            self.model.load_weights(str(weights_path))
            self.weights_loaded = True

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Expects a (H, W) or (H, W, 1) grayscale array, any size, values 0-255."""
        if image.ndim == 3:
            image = image[..., 0]
        img = tf.image.resize(image[..., np.newaxis], (IMG_SIZE, IMG_SIZE)).numpy()
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=0)  # batch dim

    def predict(self, image: np.ndarray) -> tuple[str, float, dict[str, float]]:
        batch = self.preprocess(image)
        probs = self.model.predict(batch, verbose=0)[0]
        label_idx = int(np.argmax(probs))
        per_class = {label: float(p) for label, p in zip(CLASS_LABELS, probs)}
        return CLASS_LABELS[label_idx], float(probs[label_idx]), per_class
