import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from config import generate_version_name, add_allowed_model, HYPERPARAMETERS, BASE_NAME
from prepare_data import load_and_preprocess_data


def build_and_train_model(x_train, y_train, x_test, y_test, hyperparameters):
    """
    Build, train, and log a neural network model with MLflow tracking.
    """
    version_name = generate_version_name(BASE_NAME, hyperparameters)

    with mlflow.start_run(run_name=version_name) as run:
        mlflow.log_params(hyperparameters)

        # Define a simple feedforward neural network
        model = Sequential([
            Dense(128, activation="relu", input_shape=(28 * 28,)),
            Dropout(hyperparameters["dropout_rate"]),
            Dense(64, activation="relu"),  # Added an additional dense layer for better learning
            Dropout(hyperparameters["dropout_rate"] / 2),  # Reduced dropout for this layer
            Dense(10, activation="softmax")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        history = model.fit(
            x_train, y_train,
            epochs=hyperparameters["epochs"],
            batch_size=hyperparameters["batch_size"],
            validation_data=(x_test, y_test),
            verbose=2
        )

        # Log final metrics to MLflow
        final_accuracy = history.history["val_accuracy"][-1]
        final_loss = history.history["val_loss"][-1]
        mlflow.log_metrics({"final_accuracy": final_accuracy, "final_loss": final_loss})

        # Save the model locally and log it to MLflow
        local_h5_path = generate_version_name(BASE_NAME, hyperparameters)
        model.save(local_h5_path)
        add_allowed_model(local_h5_path)
        mlflow.log_artifact(local_h5_path, artifact_path="model_h5")

        print(f"Model training completed. Validation Accuracy: {final_accuracy:.2f}")
        return model


if __name__ == "__main__":
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_data(visualize=False)

    # Train the model with the specified hyperparameters
    model = build_and_train_model(x_train, y_train, x_test, y_test, HYPERPARAMETERS)
