# Image Classification Using a Trained Model
# This script loads a trained image classification model and makes predictions on custom images.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Load the trained model (replace 'trained_model.h5' with your model file)
model = tf.keras.models.load_model('trained_model.h5')

# Define a function to preprocess an input image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return image

# Define a function to make predictions
def predict_animal(image_path, model):
    input_image = preprocess_image(image_path)
    input_image = np.expand_dims(input_image, axis=0)
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions[0])
    return predicted_class_index

# Define a function to display predictions for multiple images
def display_predictions(image_paths, model, categories):
    for image_path in image_paths:
        predicted_class_index = predict_animal(image_path, model)
        predicted_class_label = categories[predicted_class_index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title('Predicted: ' + predicted_class_label)
        plt.show()

# Specify the directory containing subdirectories (categories)
data_dir = r'your_dataset_path_here'  # Replace with your dataset path

# Use os.listdir to get a list of subdirectories (categories) in data_dir
categories = [category for category in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, category))]

# List of custom image paths (replace with your custom image paths)
custom_image_paths = [
    r'path_to_custom_image_1.jpg',
    r'path_to_custom_image_2.jpg'
]

# Display predictions for each custom image
display_predictions(custom_image_paths, model, categories)
