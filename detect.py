# Image Classifier Evaluation
# This script loads a trained image classification model, evaluates its performance
# on a test dataset, and displays sample images with actual and predicted labels.

import os
from util import load_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Specify your data directory (replace with your dataset path)
data_dir = r'your_dataset_path_here'

# Use os.listdir to get a list of subdirectories (categories) in data_dir
categories = [category for category in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, category))]

# Load data and split into train and test sets
(feature, labels) = load_data()
x_train, x_test, y_train, y_test = train_test_split(feature, labels, test_size=0.1)

# Load the trained model (replace 'trained_model.h5' with your model file)
model = tf.keras.models.load_model('trained_model.h5')

# Get predictions
predictions = model.predict(x_test)

# Initialize variables for accuracy calculation
correct_predictions = 0
total_predictions = len(y_test)

# Plot images with actual and predicted labels, and calculate accuracy
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[i])
    actual_label = categories[y_test[i]]
    predicted_label = categories[np.argmax(predictions[i])]
    is_correct = actual_label == predicted_label
    if is_correct:
        correct_predictions += 1
    plt.xlabel(f'Actual: {actual_label}\nPredicted: {predicted_label}', fontsize=10)
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()

# Calculate and display accuracy
accuracy = (correct_predictions / total_predictions) * 100
print(f'Test accuracy: {accuracy:.2f}%')
