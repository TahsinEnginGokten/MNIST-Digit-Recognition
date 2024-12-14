from load_data import load_and_visualize_data

def preprocess_data(x_train, x_test):
    # Normalize pixel values (scale between 0 and 1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Flatten the images into 1D arrays
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    return x_train, x_test

# Load data from load_data.py
x_train, y_train, x_test, y_test = load_and_visualize_data()

# Preprocess data
x_train, x_test = preprocess_data(x_train, x_test)
