from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from prepare_data import x_train, y_train, x_test, y_test

def build_and_train_model(x_train, y_train, x_test, y_test):
    # Create a simple neural network
    model = Sequential([
        Dense(128, activation="relu", input_shape=(28 * 28,)),  # Hidden layer
        Dense(10, activation="softmax")  # Output layer for 10 digits
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    return model

# Build and train the model
model = build_and_train_model(x_train, y_train, x_test, y_test)
