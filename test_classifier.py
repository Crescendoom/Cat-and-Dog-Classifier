import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from classifier import predict_image

class TestPredictImage(unittest.TestCase):
    @patch("classifier.filedialog.askopenfilename")
    @patch("classifier.tf.keras.utils.load_img")
    @patch("classifier.tf.keras.utils.img_to_array")
    @patch("classifier.model.predict")
    @patch("classifier.plt.imshow")  # Add this patch
    @patch("classifier.plt.show")   # Add this patch
    def test_predict_image(self, mock_show, mock_imshow, mock_predict, mock_img_to_array, mock_load_img, mock_askopenfilename):
        # Mock the file dialog to return a fake file path
        mock_askopenfilename.return_value = "fake_image.jpg"
        
        # Create a proper dummy image for display
        dummy_image = np.zeros((224, 224, 3))
        
        # Mock the image loading and conversion functions
        mock_load_img.return_value = dummy_image  # Use actual numpy array instead of MagicMock
        mock_img_to_array.return_value = dummy_image
        
        # Mock the model's prediction
        mock_predict.return_value = [[0.8]]  # Simulate a "Dog" prediction
        
        # Call the function
        with patch("classifier.np.expand_dims", return_value=np.zeros((1, 224, 224, 3))):
            predict_image()
        
        # Assertions
        mock_askopenfilename.assert_called_once()
        mock_load_img.assert_called_once_with("fake_image.jpg", target_size=(224, 224))
        mock_predict.assert_called_once()
        mock_imshow.assert_called_once()
        mock_show.assert_called_once()

    @patch("classifier.filedialog.askopenfilename")
    def test_no_file_selected(self, mock_askopenfilename):
        # Mock the file dialog to simulate no file selected
        mock_askopenfilename.return_value = ""
        
        # Call the function
        predict_image()
        
        # Assertions
        mock_askopenfilename.assert_called_once()

if __name__ == "__main__":
    unittest.main()