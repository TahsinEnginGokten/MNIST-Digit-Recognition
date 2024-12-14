from train_model import model
from prepare_data import x_test, y_test

def evaluate_model(model, x_test, y_test):
    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Evaluate the model
evaluate_model(model, x_test, y_test)
