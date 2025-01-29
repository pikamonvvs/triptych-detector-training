import os

import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical


def preprocess_images(folder_0, folder_1, divisor):
    """
    Process images from two folders, resize them, extract RGB values, and return as numpy arrays.

    Args:
        folder_0 (str): Path to the folder containing images with label 0.
        folder_1 (str): Path to the folder containing images with label 1.
        divisor (int): The factor by which the image dimensions will be divided.

    Returns:
        tuple: (X, y) where X is a numpy array of image data and y is a numpy array of labels.
    """
    X, y = [], []

    def process_folder(folder, label):
        for filename in os.listdir(folder):
            if filename.endswith(".png"):
                img_path = os.path.join(folder, filename)
                img = load_img(img_path)
                width, height = img.size
                img = img.resize((width // divisor, height // divisor))  # Resize image
                img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
                X.append(img_array)
                y.append(label)

    process_folder(folder_0, 0)  # Process images with label 0
    process_folder(folder_1, 1)  # Process images with label 1

    return np.array(X), np.array(y)


def build_model(input_shape):
    """
    Build and return a simple CNN model for binary classification.

    Args:
        input_shape (tuple): Shape of the input images.

    Returns:
        model: A compiled Keras model.
    """
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(2, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# Example usage
if __name__ == "__main__":
    folder_0 = input("Enter path for label 0 images: ")
    folder_1 = input("Enter path for label 1 images: ")
    divisor = int(input("Enter divisor for resizing: "))

    X, y = preprocess_images(folder_0, folder_1, divisor)
    y = to_categorical(y, num_classes=2)

    model = build_model(input_shape=X.shape[1:])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    # Set path to save the model
    model_save_path = "trained_model.h5"

    # Save model
    model.save(model_save_path)
    print(f"Model has been saved to {model_save_path}")
