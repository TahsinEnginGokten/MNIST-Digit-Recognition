# load_data.py
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def load_and_visualize_data():
    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Print dataset shapes
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Visualize a few samples
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(x_train[i], cmap="gray")
        plt.title(f"Label: {y_train[i]}")
        plt.axis("off")
    plt.show()

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    load_and_visualize_data()
