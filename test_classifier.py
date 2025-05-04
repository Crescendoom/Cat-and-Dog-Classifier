import unittest
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
from classifier import target_size
import shutil

model_path = os.path.join(os.path.dirname(__file__), "catAndDog_BinaryClassifier.keras")
model = load_model(model_path)
class TestClassifierWithRealImages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Check if test directories exist, if not create them
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

    def setUp(self):
        # Get a few sample images
        self.cat_images = [os.path.join(self.cat_dir, img) for img in os.listdir(self.cat_dir)[:3]]
        self.dog_images = [os.path.join(self.dog_dir, img) for img in os.listdir(self.dog_dir)[:3]]

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