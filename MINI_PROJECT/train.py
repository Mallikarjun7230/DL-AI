"""
Training Pipeline for MNIST Neural Network
Trains CNN on MNIST dataset, records metrics, and saves the trained model.
"""

import json
import os
import time
import numpy as np
import tensorflow as tf
from model import build_mnist_cnn


def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(base_dir, "mnist_cnn.keras")
    metrics_save_path = os.path.join(base_dir, "training_metrics.json")
    samples_save_path = os.path.join(base_dir, "mnist_test_samples.npz")

    print("=" * 60)
    print("MNIST CNN Training Pipeline Initiated")
    print("=" * 60)

    # 1. Load MNIST Dataset
    print("\n[1/4] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(f"  Training samples:   {x_train.shape[0]:,}")
    print(f"  Testing samples:    {x_test.shape[0]:,}")

    # 2. Normalize and Reshape
    print("\n[2/4] Normalizing & shaping input tensors...")
    x_train = (x_train.astype(np.float32) / 255.0)[..., np.newaxis]
    x_test = (x_test.astype(np.float32) / 255.0)[..., np.newaxis]

    # Save a small subset of test samples for web interface live testing
    # Store 50 random test samples with their true labels
    sample_indices = np.random.RandomState(42).choice(len(x_test), 50, replace=False)
    np.savez_compressed(
        samples_save_path,
        images=(x_test[sample_indices] * 255.0).astype(np.uint8).squeeze(-1),
        labels=y_test[sample_indices]
    )
    print(f"  Saved 50 test samples for live UI demos: {samples_save_path}")

    # 3. Build & Compile Model
    print("\n[3/4] Building CNN architecture...")
    model = build_mnist_cnn()
    total_params = model.count_params()
    print(f"  Total trainable parameters: {total_params:,}")

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 4. Train Model
    print("\n[4/4] Training for 6 epochs...")
    start_time = time.time()
    history = model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=6,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    training_duration = round(time.time() - start_time, 2)

    # Final Evaluation
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("\n" + "=" * 60)
    print(f"Training Complete in {training_duration}s!")
    print(f"  Final Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"  Final Test Loss:     {test_loss:.4f}")
    print("=" * 60)

    # Save trained model
    model.save(model_save_path)
    print(f"\nModel saved successfully: {model_save_path}")

    # Save metrics metadata
    metrics = {
        "model_name": "Deep CNN (Conv2D x 4 + BatchNorm + Dropout)",
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "test_accuracy": round(float(test_accuracy), 5),
        "test_accuracy_percentage": f"{test_accuracy * 100:.2f}%",
        "test_loss": round(float(test_loss), 5),
        "total_parameters": int(total_params),
        "epochs_trained": len(history.history['loss']),
        "training_time_seconds": training_duration,
        "history": {
            "accuracy": [round(float(v), 5) for v in history.history['accuracy']],
            "val_accuracy": [round(float(v), 5) for v in history.history['val_accuracy']],
            "loss": [round(float(v), 5) for v in history.history['loss']],
            "val_loss": [round(float(v), 5) for v in history.history['val_loss']],
        }
    }

    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics metadata saved: {metrics_save_path}")


if __name__ == "__main__":
    train()
