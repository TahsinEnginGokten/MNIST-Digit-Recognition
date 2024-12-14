from flask import Flask, request, jsonify, send_from_directory  # Added send_from_directory for static file serving
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.datasets import mnist  # Importing MNIST dataset

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("models/digit_recognition_model.h5")

# Directory to save MNIST sample images
SAMPLE_DIR = "mnist_samples"
os.makedirs(SAMPLE_DIR, exist_ok=True)

def generate_sample_images():
    """
    Generate 10 sample images from the MNIST dataset and save them to a directory.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    for i in range(10):
        image = Image.fromarray(x_test[i])  # Convert the NumPy array to a PIL image
        label = y_test[i]  # Get the corresponding label
        image_path = os.path.join(SAMPLE_DIR, f"mnist_digit_{i}_label_{label}.png")
        image.save(image_path)

# Generate sample images on startup
generate_sample_images()

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the digit in an uploaded image using the trained model.
    """
    try:
        # Get the file from the request
        file = request.files.get("file")

        # Ensure the file is provided and is an image
        if not file or not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Invalid file format. Please upload a .png, .jpg, or .jpeg image."}), 400

        # Open the image and preprocess
        image = Image.open(io.BytesIO(file.read())).convert("L").resize((28, 28))
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 28 * 28)

        # Make prediction
        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction)

        # Return the result
        return jsonify({"digit": int(predicted_digit)})

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": str(e)}), 500
    

@app.route("/predict_sample", methods=["GET"])
def predict_sample():
    """
    Predict the digit for a predefined sample image from the MNIST dataset.
    """
    try:
        # Load the MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Select a sample image (for example, the 0th image from the test set)
        sample_image = x_test[0]

        # Preprocess the image (resize, normalize, reshape)
        image_array = sample_image / 255.0
        image_array = image_array.reshape(1, 28 * 28)

        # Predict the digit
        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction)

        # Return the result
        return jsonify({"digit": int(predicted_digit)})

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": str(e)}), 500


@app.route("/get_samples", methods=["GET"])
def get_samples():
    """
    Get a list of all generated MNIST sample images.
    """
    try:
        # List all images in the sample directory
        samples = os.listdir(SAMPLE_DIR)
        sample_urls = [f"http://127.0.0.1:5000/mnist_samples/{sample}" for sample in samples]
        return jsonify({"sample_urls": sample_urls})  # Return accessible URLs
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/mnist_samples/<filename>", methods=["GET"])
def serve_sample_file(filename):
    """
    Serve a specific MNIST sample image file.
    """
    try:
        return send_from_directory(SAMPLE_DIR, filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    app.run(debug=True)
