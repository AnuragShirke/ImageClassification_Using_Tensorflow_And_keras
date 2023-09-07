# Image Classifier Using Transfer Learning (MobileNetV2)
# This script loads a pre-trained MobileNetV2 model, removes the top classification layer,
# adds a custom classification head, and trains the model on a custom image dataset.
# It then evaluates the model's performance and saves the trained model to a file.

import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from util import load_data

# Specify your data directory (replace with your dataset path)
data_dir = r'your_dataset_path_here'

# Use os.listdir to get a list of subdirectories (categories) in data_dir
categories = [category for category in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, category))]

# Load and preprocess data
(feature, labels) = load_data()
x_train, x_test, y_train, y_test = train_test_split(feature, labels, test_size=0.1)

# Load pre-trained MobileNetV2 model without the top classification layer
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create your own custom classification head
flatten_layer = tf.keras.layers.Flatten()(base_model.output)
dense_layer = tf.keras.layers.Dense(512, activation='relu')(flatten_layer)
# Modify the output layer to have the correct number of units (82)
output_layer = tf.keras.layers.Dense(len(categories), activation='softmax')(dense_layer)

# Create the final model
model = tf.keras.Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=100, epochs=10)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)

# Save the trained model to a file
model.save('trained_model.h5')
