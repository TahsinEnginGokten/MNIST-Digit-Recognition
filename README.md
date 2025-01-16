# Flask-Based MNIST Digit Recognition API

This project provides a Flask API for recognizing handwritten digits using a trained TensorFlow model. It supports dynamic model loading, predictions from uploaded images, and serves preprocessed MNIST dataset samples for testing and experimentation.

---

## Features

- **Dynamic Model Loading**: Load different trained models on-the-fly via an API endpoint.
- **Prediction**: Upload images to predict digits, or use predefined MNIST samples.
- **Model Training and Evaluation**: Train, evaluate, and save TensorFlow models with versioned names.
- **Healthcheck**: Verify API health with a simple endpoint.
- **Visualization**: View sample MNIST data to understand the dataset.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TahsinEnginGokten/mnist-digit-recognition-api.git
   cd mnist-digit-recognition-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `models` directory exists and contains the default model (`digit_recognition_model.h5`).

---

## Usage

### Run the API

Start the Flask application:
```bash
python app.py
```

### API Endpoints

1. **Healthcheck**
   - **GET** `/healthcheck`
   - Returns API status.

2. **Load Model**
   - **POST** `/load_model/<model_name>`
   - Dynamically loads a model by name (e.g., `digit_recognition_v1.h5`).

3. **Predict Uploaded Image**
   - **POST** `/predict`
   - Upload a grayscale image of a digit (28x28 pixels) in PNG/JPG format.

4. **Get MNIST Sample Image**
   - **GET** `/mnist_sample/<int:index>`
   - Returns an MNIST dataset sample image by index.

5. **Predict MNIST Sample**
   - **GET** `/predict_sample/<int:index>`
   - Predicts the digit from the MNIST dataset sample by index.

---

## Model Training

To train a new model:
1. Configure hyperparameters in `config.py`.
2. Run the training pipeline:
   ```bash
   python main.py
   ```
3. Versioned models will be saved in the `models` directory.

---

## Configuration

Key configurations are in `config.py`:
- Default model path: `DEFAULT_MODEL`
- Model hyperparameters: `HYPERPARAMETERS`
- Allowed models: `ALLOWED_MODELS`

---

## Dependencies

- Flask
- TensorFlow
- NumPy
- Pillow
- MLflow (for model tracking)

---

## Example Prediction

### Predict with Uploaded Image
1. Save a test image as `test_image.png`.
2. Use `curl` to test the `/predict` endpoint:
   ```bash
   curl -X POST -F "file=@test_image.png" http://127.0.0.1:5000/predict
   ```

---

## Future Improvements

- Add more complex models and datasets.
- Integrate CI/CD for automated deployment.
- Expand support for additional image formats and preprocessing.

---

## Contributing

Contributions are welcome! Fork the repository and submit a pull request with your improvements.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
