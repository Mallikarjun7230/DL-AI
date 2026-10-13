"""
MNIST Neural Network Architecture
High-accuracy Convolutional Neural Network (CNN) for handwritten digit classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_mnist_cnn() -> models.Sequential:
    """
    Builds and compiles an optimized CNN model for MNIST digit recognition.
    Input shape: (28, 28, 1)
    Output shape: (10,) softmax probabilities
    """
    model = models.Sequential([
        # Block 1
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Dense Classification Head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax', name='digits_output')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    net = build_mnist_cnn()
    net.summary()
