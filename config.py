### Updated config.py ###
import os
from datetime import datetime

# Define the directory where all models will be saved/loaded
MODELS_DIR = "models/"

# Ensure the models directory exists
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Base model name (used for default and versioned models)
BASE_NAME = "digit_recognition"

# Default hyperparameters
HYPERPARAMETERS = {
    "epochs": 10,  # Increased epochs for better training
    "batch_size": 64,  # Adjusted batch size
    "learning_rate": 0.001,  # Common learning rate for Adam optimizer
    "dropout_rate": 0.3  # Reduced dropout for a balance between regularization and learning
}

# Default model path (for production or standard use)
DEFAULT_MODEL = os.path.join(MODELS_DIR, f"{BASE_NAME}_model.h5")

# List of currently allowed model paths for Flask
ALLOWED_MODELS = [DEFAULT_MODEL]

# Function to generate versioned model names
def generate_version_name(base_name, hyperparameters):
    """
    Generate a unique version name for the model based on hyperparameters and timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    version_name = (
        f"{base_name}_v_epochs-{hyperparameters['epochs']}_"
        f"batch-{hyperparameters['batch_size']}_"
        f"lr-{hyperparameters['learning_rate']}_"
        f"dropout-{hyperparameters['dropout_rate']}_{timestamp}.h5"
    )
    return os.path.join(MODELS_DIR, version_name)

# Function to dynamically add models to the allowed list
def add_allowed_model(model_path):
    """
    Add a model path to the allowed list if it exists and is not already added.
    """
    if os.path.exists(model_path) and model_path not in ALLOWED_MODELS:
        ALLOWED_MODELS.append(model_path)

# Function to view all allowed models
def view_allowed_models():
    """
    Print the list of all allowed models.
    """
    print("Allowed Models:")
    for model in ALLOWED_MODELS:
        print(model)
