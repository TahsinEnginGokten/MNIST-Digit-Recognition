import os
from config import add_allowed_model, view_allowed_models, BASE_NAME

def update_default_model(selected_model_path):
    """
    Replace the default model with a selected versioned model.
    """
    default_model_path = os.path.join("models", f"{BASE_NAME}_model.h5")
    try:
        # Check if the selected model exists
        if not os.path.exists(selected_model_path):
            raise FileNotFoundError(f"Selected model not found: {selected_model_path}")

        # Replace the default model
        os.replace(selected_model_path, default_model_path)
        add_allowed_model(default_model_path)
        print(f"Default model updated to '{default_model_path}'.")
    except Exception as e:
        print(f"Error updating default model: {e}")

if __name__ == "__main__":
    # Example usage: Update the default model
    latest_model_path = "models/digit_recognition_v_epochs-10_batch-64_lr-0.001_dropout-0.3_20250115-153600.h5"
    print("Before updating the default model:")
    view_allowed_models()

    update_default_model(latest_model_path)

    print("\nAfter updating the default model:")
    view_allowed_models()
