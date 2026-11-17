"""
RetailMind Day 6 — Reusable CNN image-classification functions.

Dataset: Fashion-MNIST loaded directly from TensorFlow/Keras.
No Kaggle download, styles.csv, or local images folder is required.

Import examples (from the RetailMind project root):
    from src.image_classification import (
        load_fashion_mnist,
        build_cnn,
        train_cnn,
        evaluate_cnn,
        save_model,
        predict_image,
        CLASS_NAMES,
    )

This file contains reusable functions only. The notebook is responsible for
explaining the workflow, calling these functions, and displaying plots.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Fashion-MNIST label names in the same order as integer labels 0 through 9.
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_fashion_mnist():
    """
    Load Fashion-MNIST from Keras and preprocess the images.

    Returns:
        (x_train, y_train), (x_test, y_test)

    Image arrays have shape:
        x_train: (60000, 28, 28, 1)
        x_test:  (10000, 28, 28, 1)

    Pixel values are converted from 0-255 to 0-1.
    Labels remain integer class IDs from 0 to 9.
    """
    (x_train, y_train), (x_test, y_test) = (
        keras.datasets.fashion_mnist.load_data()
    )

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add the channel dimension required by Conv2D for grayscale images.
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return (x_train, y_train), (x_test, y_test)


def build_cnn(input_shape=(28, 28, 1), num_classes=10, learning_rate=None):
    """
    Create and compile a CNN for Fashion-MNIST classification.

    Architecture:
        Conv2D -> MaxPooling2D -> Conv2D -> MaxPooling2D
        -> Flatten -> Dense -> Dropout -> Softmax

    Sparse categorical cross-entropy is used because labels are integer IDs,
    not one-hot encoded vectors.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="retailmind_fashion_mnist_cnn",
    )

    optimizer = (
        keras.optimizers.Adam(learning_rate=learning_rate)
        if learning_rate is not None
        else "adam"
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_cnn(
    model,
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.10,
):
    """
    Train a compiled CNN and return its Keras History object.

    Validation data is taken only from the training arrays. Keep the separate
    test set untouched until final evaluation.
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    return history


def evaluate_cnn(model, x_test, y_test):
    """
    Evaluate a trained model on held-out test data.

    Returns a dictionary containing:
        test_loss, test_accuracy, probabilities, predictions,
        and the 10-by-10 confusion matrix.
    """
    test_loss, test_accuracy = model.evaluate(
        x_test, y_test, verbose=0
    )

    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    confusion_matrix = tf.math.confusion_matrix(
        y_test,
        predictions,
        num_classes=len(CLASS_NAMES),
    ).numpy()

    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "probabilities": probabilities,
        "predictions": predictions,
        "confusion_matrix": confusion_matrix,
    }


def save_model(model, model_path):
    """
    Save a trained Keras model.

    Example:
        save_model(model, "models/retailmind_fashion_mnist_day6.keras")
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return model_path


def load_saved_model(model_path):
    """Load a previously saved Keras model."""
    return keras.models.load_model(model_path)


def predict_image(model, image):
    """
    Predict the category of one Fashion-MNIST image.

    Accepts a 28x28 or 28x28x1 image. Pixel values may be either raw
    0-255 values or already normalized 0-1 values.

    Returns:
        predicted_class_name, confidence, all_class_probabilities
    """
    image = np.asarray(image, dtype="float32")

    if image.ndim == 2:
        image = image[..., np.newaxis]

    if image.shape != (28, 28, 1):
        raise ValueError(
            "Expected image shape (28, 28) or (28, 28, 1); "
            f"received {image.shape}."
        )

    if image.max() > 1.0:
        image = image / 255.0

    probabilities = model.predict(
        np.expand_dims(image, axis=0),
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence, probabilities


def save_evaluation_metrics(results, output_path):
    """
    Save the test loss and accuracy as a JSON file.

    The NumPy arrays (predictions, probabilities and confusion matrix) are
    intentionally excluded from this compact metrics JSON.
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": "CNN - Fashion-MNIST",
        "test_loss": round(results["test_loss"], 4),
        "test_accuracy": round(results["test_accuracy"], 4),
        "number_of_classes": len(CLASS_NAMES),
    }

    output_path.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    return output_path
