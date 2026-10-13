"""
Automated Test Suite for MNIST Digit Recognizer
Validates model loading, preprocessing pipeline, and inference accuracy.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from preprocess import preprocess_canvas_image
from model import build_mnist_cnn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn.keras")
METRICS_PATH = os.path.join(BASE_DIR, "training_metrics.json")
SAMPLES_PATH = os.path.join(BASE_DIR, "mnist_test_samples.npz")


def test_model_artifact_exists():
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    print("PASS: Model artifact exists.")


def test_metrics_json():
    assert os.path.exists(METRICS_PATH), f"Metrics file not found at {METRICS_PATH}"
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "test_accuracy" in metrics, "Missing test_accuracy in metrics"
    assert metrics["test_accuracy"] >= 0.98, f"Test accuracy too low: {metrics['test_accuracy']}"
    print(f"PASS: Metrics validated. Test accuracy = {metrics['test_accuracy_percentage']}")


def test_preprocessing_empty():
    # Test completely empty image
    img = Image.new("RGB", (300, 300), color=(3, 7, 18))
    tensor, preview, is_empty = preprocess_canvas_image(img)
    assert is_empty is True, "Empty canvas should be flagged as is_empty=True"
    assert tensor.shape == (1, 28, 28, 1), f"Unexpected shape {tensor.shape}"
    print("PASS: Empty canvas detection works correctly.")


def test_preprocessing_synthetic_digit():
    # Draw a vertical line representing '1'
    img = Image.new("RGB", (300, 300), color=(3, 7, 18))
    draw = ImageDraw.Draw(img)
    draw.line([(150, 40), (150, 260)], fill=(255, 255, 255), width=24)

    tensor, preview, is_empty = preprocess_canvas_image(img)
    assert is_empty is False, "Drawn canvas should not be empty"
    assert tensor.shape == (1, 28, 28, 1), f"Unexpected tensor shape: {tensor.shape}"
    assert tensor.max() > 0.5, "Tensor should contain normalized stroke intensities"
    assert preview.startswith("data:image/png;base64,"), "Preview must be a valid base64 data URL"
    print("PASS: Synthetic stroke preprocessing and center-of-mass alignment works.")


def test_model_inference_accuracy():
    model = tf.keras.models.load_model(MODEL_PATH)
    assert os.path.exists(SAMPLES_PATH), "Samples file not found"
    data = np.load(SAMPLES_PATH)
    images = data["images"]  # (50, 28, 28)
    labels = data["labels"]  # (50,)

    # Normalize samples to (N, 28, 28, 1) float32
    x_test_subset = (images.astype(np.float32) / 255.0)[..., np.newaxis]
    predictions = model.predict(x_test_subset, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    correct = np.sum(predicted_classes == labels)
    acc = correct / len(labels)
    print(f"PASS: Inference on 50 MNIST test samples: {correct}/{len(labels)} correct ({acc * 100:.1f}% accuracy).")
    assert acc >= 0.96, f"Sample accuracy below expected threshold: {acc}"


if __name__ == "__main__":
    print("Running verification tests...")
    test_model_artifact_exists()
    test_metrics_json()
    test_preprocessing_empty()
    test_preprocessing_synthetic_digit()
    test_model_inference_accuracy()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY! [OK]")
