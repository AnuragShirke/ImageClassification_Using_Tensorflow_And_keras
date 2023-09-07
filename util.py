# Image Dataset Preprocessing
# This script preprocesses an image dataset containing various categories(Eg.Animals(Lions.Tigers,etc),Flowers(Tulips,Roses,etc),Etc).
# It resizes the images, converts them to the RGB color space, and saves the data as a pickle file.
import os
import numpy as np
import cv2
import pickle

# Specify the path to your dataset directory
data_dir = r'your_dataset_path_here'
categories = []

# Use os.listdir to get a list of subdirectories (categories) in data_dir
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    if os.path.isdir(category_path):
        categories.append(category)

data = []

def make_data():
    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)
        
        # Print the current category being preprocessed
        print(f"Preprocessing category: {category}")

        for img_name in os.listdir(path):
            image_path = os.path.join(path, img_name)
            image = cv2.imread(image_path)

            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))

                image = np.array(image, dtype=np.float32)
                data.append([image, label])

            except Exception as e:
                pass

    print(len(data))

    with open('data.pickle', 'wb') as pik:
        pickle.dump(data, pik)

make_data()

def load_data():
    with open('data.pickle', 'rb') as pik:
        data = pickle.load(pik)

    np.random.shuffle(data)

    features = []
    labels = []

    for img, label in data:
        features.append(img)
        labels.append(label)

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels)

    features = features / 255.0

    return [features, labels]

features, labels = load_data()

