import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
from tensorflow.keras.models import load_model # type: ignore

model = load_model("path/to/catAndDog_BinaryClassifier.keras")
target_size = (224, 224)

def predict_image():
    root = Tk()
    root.withdraw()

    img_path = filedialog.askopenfilename(title='Select an image', filetypes=[("Image files", "*.jpg;*.png")])
    
    if not img_path:
        print("No file selected.")
        return

    img = tf.keras.utils.load_img(img_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    label = 'Dog' if prediction[0][0] > 0.5 else 'Cat'

    plt.imshow(img)
    plt.title(f"Predicted: {label}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    predict_image()
