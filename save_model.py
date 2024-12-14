import os
from train_model import model

def save_model(model):
    # Create the models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model.save('models/digit_recognition_model.h5')
    print("Model saved as 'models/digit_recognition_model.h5'")

# Save the trained model
save_model(model)
