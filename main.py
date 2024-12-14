# main.py
from load_data import load_and_visualize_data
from prepare_data import preprocess_data
from train_model import build_and_train_model
from evaluate_model import evaluate_model
from save_model import save_model

def main():
    # Part 1: Load and visualize the data
    x_train, y_train, x_test, y_test = load_and_visualize_data()

    # Part 2: Prepare the data (normalize and flatten)
    x_train, x_test = preprocess_data(x_train, x_test)

    # Part 3: Build and train the model
    model = build_and_train_model(x_train, y_train, x_test, y_test)

    # Part 4: Evaluate the model
    evaluate_model(model, x_test, y_test)

    # Part 5: Save the trained model
    save_model(model)

if __name__ == "__main__":
    main()