# Cat and Dog Image Classifier

A deep learning model that classifies images as either cats or dogs using TensorFlow and MobileNetV2.

<a href="https://cat-and-dog-classifier-d5qqukhnle8rqmdsvrjqud.streamlit.app/" target="_blank">asdfadfa
  <img src="https://img.shields.io/badge/Try_the_App-Cat_and_Dog_Classifier-FF4B4B?style=for-the-badge&logo=streamlit" alt="Try the App">
</a>

## Requirements

- Python 3.10
- TensorFlow 2.12.0
- Numpy, Matplotlib, Pillow 
- See requirements.txt for all dependencies

## Setup

1. Clone this repository
2. Create a virtual environment: 
    - python -m venv .venv
3. Activate the environment:
    - Windows: activate
    - Linux/Mac: source .venv/bin/activate
4. Install dependencies: 
    - pip install -r requirements.txt
5. Download the dataset with instructions below (if you want to train a model yourself)

## Dataset 

Download the cats and dogs dataset from Kaggle https://www.kaggle.com/code/robikscube/working-with-image-data-in-python/input, and organize it as follows:
- Dataset/
    - training_set/
        - cats/      ~4,000 cat images
        - dogs/      ~4,000 cat images
    - validation_set/
        - cats       ~1,000 cat images
        - dogs       ~1,000 dog images

## Usage

1. First, train the model (if not already trained): python train.py (uncomment the lines needed)
    - This creates the catAndDog_BinaryClassifier.keras file.

2. Run the classifier: 
    - python classifier.py

3. Select an image when prompted, and the classifier will display its prediction.

## Testing

Run the automated tests to verify the classifier works correctly: 
- python -m unittest
    
## Project Structure
 - main.py - Main script for classifying images in web use
 - classifier.py - Main script for classifying images in desktop use
 - train.py - Script for training the model
 - test_classifier.py - Unit tests
 - requirements.txt - Python dependencies
 - catAndDog_BinaryClassifier.keras - Pre-trained model file

## Acknowledgement 
Project idea from Rob Mulla https://youtu.be/kSqxn6zGE0c?si=GkTxVZJoowRXr2yL