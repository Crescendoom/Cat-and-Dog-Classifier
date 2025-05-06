import unittest
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
from classifier import target_size
import random
import shutil

model_path = os.path.join(os.path.dirname(__file__), "catAndDog_BinaryClassifier.keras")
model = load_model(model_path)

class TestClassifierWithDatasetImages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define dataset paths
        cls.base_dir = "Dataset"
        cls.validation_dir = os.path.join(cls.base_dir, "validation_set")
        cls.test_dir = os.path.join(cls.base_dir, "training_set")
        
        # Define image source paths to try in order of preference
        cls.cat_source_paths = [
            os.path.join(cls.validation_dir, "cats"),  # First try validation cats
            os.path.join(cls.test_dir, "cats"),    # Then try training cats
        ]
        
        cls.dog_source_paths = [
            os.path.join(cls.validation_dir, "dogs"),  # First try validation dogs
            os.path.join(cls.test_dir, "dogs"),    # Then try training dogs
        ]
        
        # Find the first valid cat and dog directories
        cls.cat_dir = next((path for path in cls.cat_source_paths 
                           if os.path.exists(path) and len(os.listdir(path)) > 0), None)
        cls.dog_dir = next((path for path in cls.dog_source_paths 
                           if os.path.exists(path) and len(os.listdir(path)) > 0), None)
        
        # If no valid directories are found, fall back to test_set
        if not cls.cat_dir or not cls.dog_dir:
            print("No suitable dataset images found. Falling back to test_set directory.")
            cls.test_dir = os.path.join("test_set")
            cls.cat_dir = os.path.join(cls.test_dir, "cats")
            cls.dog_dir = os.path.join(cls.test_dir, "dogs")
            
            if not os.path.exists(cls.test_dir):
                os.makedirs(cls.test_dir, exist_ok=True)
                os.makedirs(cls.cat_dir, exist_ok=True)
                os.makedirs(cls.dog_dir, exist_ok=True)
                print(f"Created test directories at {cls.test_dir}")
                print("Please add cat images to test_set/cats/ and dog images to test_set/dogs/")
                print("Then run the test again.")
                raise unittest.SkipTest("Test directories created. Add images and run again.")
            
            # Check if there are images in the directories
            if len(os.listdir(cls.cat_dir)) == 0 or len(os.listdir(cls.dog_dir)) == 0:
                print("No images found in test directories.")
                print("Please add cat images to test_set/cats/ and dog images to test_set/dogs/")
                raise unittest.SkipTest("No test images found. Add images and run again.")
        else:
            print(f"Using cat images from: {cls.cat_dir}")
            print(f"Using dog images from: {cls.dog_dir}")

    def setUp(self):
        # Get a few sample images (up to 3)
        cat_all_images = os.listdir(self.cat_dir)
        dog_all_images = os.listdir(self.dog_dir)
        
        # Randomly select up to 3 images from each category
        cat_sample_count = min(3, len(cat_all_images))
        dog_sample_count = min(3, len(dog_all_images))
        
        self.cat_images = [os.path.join(self.cat_dir, img) for img in random.sample(cat_all_images, cat_sample_count)]
        self.dog_images = [os.path.join(self.dog_dir, img) for img in random.sample(dog_all_images, dog_sample_count)]

    def predict_single_image(self, img_path):
        """Function that replicates the logic in predict_image but accepts a filepath directly"""
        img = tf.keras.utils.load_img(img_path, target_size=target_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array, verbose=0)
        label = 'Dog' if prediction[0][0] > 0.5 else 'Cat'
        
        return label, prediction[0][0]

    def test_cat_images(self):
        """Test that cat images are correctly classified"""
        for img_path in self.cat_images:
            label, confidence = self.predict_single_image(img_path)
            print(f"Image: {os.path.basename(img_path)}, Prediction: {label}, Confidence: {confidence:.4f}")
            self.assertEqual(label, 'Cat', f"Failed to classify cat image {os.path.basename(img_path)}")

    def test_dog_images(self):
        """Test that dog images are correctly classified"""
        for img_path in self.dog_images:
            label, confidence = self.predict_single_image(img_path)
            print(f"Image: {os.path.basename(img_path)}, Prediction: {label}, Confidence: {confidence:.4f}")
            self.assertEqual(label, 'Dog', f"Failed to classify dog image {os.path.basename(img_path)}")

if __name__ == "__main__":
    unittest.main()