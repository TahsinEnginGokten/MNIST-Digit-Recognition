from prepare_data import load_and_preprocess_data  # Centralized function
from train_model import build_and_train_model
from evaluate_model import evaluate_model
from save_model import save_model
from config import HYPERPARAMETERS, BASE_NAME, add_allowed_model, generate_version_name

def main():
    """
    Main pipeline for training, saving, and evaluating the model.
    """
    # Part 1: Load, visualize, and preprocess the data
    x_train, y_train, x_test, y_test = load_and_preprocess_data(visualize=True)  # Toggle visualization as needed

    # Part 2: Build and train the model
    model = build_and_train_model(
        x_train,
        y_train,
        x_test,
        y_test,
        HYPERPARAMETERS  # Pass hyperparameters from the centralized config
    )

    # Generate the versioned model name
    model_path = generate_version_name(BASE_NAME, HYPERPARAMETERS)

    # Part 3: Save the trained model
    save_model(model, HYPERPARAMETERS)  # Save the model with meaningful version naming
    add_allowed_model(model_path)  # Add the model to the allowed list

    # Part 4: Evaluate the model
    evaluate_model(model_path, x_test, y_test)  # Use the dynamically generated model path

if __name__ == "__main__":
    main()
