from load_data import load_and_visualize_data

def preprocess_data(x_train, x_test):
    """
    Normalize pixel values and flatten the data for model training.
    """
    # Normalize pixel values (scale between 0 and 1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Flatten the images into 1D arrays
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    return x_train, x_test

def load_and_preprocess_data(visualize=False):
    """
    Load and preprocess the MNIST dataset, with optional visualization.
    """
    x_train, y_train, x_test, y_test = load_and_visualize_data(visualize=visualize)
    x_train, x_test = preprocess_data(x_train, x_test)
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_and_preprocess_data(visualize=True)
    print("Data preprocessing complete.")