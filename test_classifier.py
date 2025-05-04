import unittest
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from classifier import model, target_size

class TestClassifierWithRealImages(unittest.TestCase):
    def setUp(self):
        # Path to test images
        self.test_dir = os.path.join("test_set")
        self.cat_dir = os.path.join(self.test_dir, "cats")
        self.dog_dir = os.path.join(self.test_dir, "dogs")
        
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