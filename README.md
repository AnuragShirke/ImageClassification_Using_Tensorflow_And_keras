# ImageClassification_Using_Tensorflow_And_keras
This repository contains Python scripts for training and using an image classifier. The classifier uses transfer learning with a pre-trained model to classify images ofany objects or things like animals,flowers,etc into various categories.

## Prerequisites_And_Usage

Before using the provided scripts, ensure you have the following prerequisites installed:

- Python (3.6 or higher)
- TensorFlow (2.x)
- OpenCV (cv2)
- NumPy
- Matplotlib

You can install these dependencies using pip:

```bash
pip install tensorflow opencv-python numpy matplotlib

## Usage 
Follow these steps to train the model, make predictions, and use custom images:

1. **Data Preparation:**

   - Organize your image dataset into subdirectories (categories) in a folder.
   - Modify the `data_dir` variable in `util.py` to point to your dataset directory.

2. **Data Preprocessing:**

   - Run `util.py` to preprocess your dataset. This script will convert images to the required format and save the data as a pickle file.
   
     ```bash
     python util.py
     ```

3. **Model Training:**

   - Train the image classifier using `classifier.py`. This script loads a pre-trained MobileNetV2 model, adds a custom classification head, and trains it on your dataset.
   
     ```bash
     python classifier.py
     ```

4. **Evaluate Model:**

   - After training, the model's performance is evaluated on a test set. You can view predictions on sample images and check the test accuracy.
   
     ```bash
     python detect.py
     ```

5. **Custom Predictions:**

   - Use the `custom.py` script to make predictions on custom images. Specify the paths to your custom images in the `custom_image_paths` list.
   
     ```bash
     python custom.py
     ```
