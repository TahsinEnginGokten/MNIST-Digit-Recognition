### Updated evaluate_model.py ###
import mlflow
from tensorflow.keras.models import load_model
from config import ALLOWED_MODELS, add_allowed_model, DEFAULT_MODEL
from prepare_data import load_and_preprocess_data


def evaluate_model(model_path, x_test, y_test):
    """
    Evaluate the specified model and log results with MLflow.
    """
    if model_path not in ALLOWED_MODELS:
        print(f"Skipping unauthorized model: {model_path}")
        return

    try:
        model = load_model(model_path)
        print(f"Evaluating model: {model_path}")

        # Start an MLflow run
        with mlflow.start_run(run_name=f"evaluate_{model_path}"):
            test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
            mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")

    except Exception as e:
        print(f"Error during evaluation of {model_path}: {e}")


if __name__ == "__main__":
    # Load test data
    _, _, x_test, y_test = load_and_preprocess_data(visualize=False)

    # Evaluate each allowed model
    for model_path in ALLOWED_MODELS:
        evaluate_model(model_path, x_test, y_test)
