from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist
from config import DEFAULT_MODEL, ALLOWED_MODELS, add_allowed_model
import os
import io

app = Flask(__name__)

# Load the default model at startup
if os.path.exists(DEFAULT_MODEL):
    model = tf.keras.models.load_model(DEFAULT_MODEL)
    add_allowed_model(DEFAULT_MODEL)  # Ensure the default model is allowed
    print(f"Default model loaded: {DEFAULT_MODEL}")  # Debugging
else:
    raise FileNotFoundError(f"Default model not found at {DEFAULT_MODEL}")

@app.route("/load_model/<model_name>", methods=["POST"])
def load_model_endpoint(model_name):
    """
    Dynamically load a specified model by its name.
    """
    global model
    try:
        model_path = f"models/{model_name}.h5"
        print(f"Requested model path: {model_path}")  # Debugging
        print(f"Allowed Models: {ALLOWED_MODELS}")   # Debugging

        if model_path in ALLOWED_MODELS and os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return jsonify({"status": f"Model {model_name} loaded successfully."})
        else:
            return jsonify({"error": "Model not allowed or does not exist."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the digit in an uploaded image using the trained model.
    """
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded."}), 400

        try:
            image = Image.open(io.BytesIO(file.read())).convert("L").resize((28, 28))
            image_array = np.array(image) / 255.0
            image_array = image_array.reshape(1, 28 * 28)
        except Exception:
            return jsonify({"error": "Invalid image file. Ensure it's a valid .png, .jpg, or .jpeg image."}), 400

        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit] * 100  # Convert to percentage
        return jsonify({"digit": int(predicted_digit), "confidence": f"{confidence:.2f}%"})  # Format as percentage

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    """
    Healthcheck endpoint to ensure the API is running.
    """
    return jsonify({"status": "API is running."})

@app.route("/mnist_sample/<int:index>", methods=["GET"])
def get_sample_image(index):
    """
    Serve an MNIST sample image dynamically by index.
    """
    try:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if index < 0 or index >= len(x_test):
            return jsonify({"error": "Invalid index. Choose a value between 0 and 9999."}), 400

        sample_image = Image.fromarray(x_test[index])
        buf = io.BytesIO()
        sample_image.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_sample/<int:index>", methods=["GET"])
def predict_sample(index):
    """
    Predict the digit for an MNIST sample image by index.
    """
    try:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        if index < 0 or index >= len(x_test):
            return jsonify({"error": "Invalid index. Choose a value between 0 and 9999."}), 400

        sample_image = x_test[index] / 255.0
        sample_image = sample_image.reshape(1, 28 * 28)

        prediction = model.predict(sample_image)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit] * 100  # Convert to percentage
        return jsonify({"digit": int(predicted_digit), "confidence": f"{confidence:.2f}%"})  # Format as percentage


    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the app (debug mode should only be used during development)
    print(f"Allowed Models at Startup: {ALLOWED_MODELS}")  # Debugging
    app.run(debug=True)
