import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore

st.set_page_config(page_title="Cat and Dog Classifier", layout="wide")

@st.cache_resource
def load_classifier_model():
    return load_model("catAndDog_BinaryClassifier.keras")

def main():
    st.title("Cat and Dog Classifier")
    st.write("Upload an image to check if it's a cat or a dog")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("This app uses a pre-trained deep learning model to classify images as cats or dogs.")
    st.sidebar.info("""
                    **Framework:** TensorFlow  
                    **Model:** MobileNetV2
                    """)
    
    st.sidebar.header("Creator")
    st.sidebar.info("""
                    **John Laurence Hernandez**  
                    [GitHub](https://github.com/Crescendoom)
                    """)

    # Load model
    model = load_classifier_model()

    # If you want an error handling for loading the model, use this: instead of "model = load_classifier_model()" only
    # try:
    #     model = load_model("catAndDog_BinaryClassifier.keras")        
    #     model_load_state = st.success("Model loaded successfully!")
    # except Exception as e:
    #     st.error(f"Error loading model: {e}")
    #     st.info("Make sure the model file 'catAndDog_BinaryClassifier.keras' exists in your project directory.")
    #     return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image to upload.", type=["jpg", "jpeg", "png"])
    
    # Process uploaded file
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        
        # Process image for prediction
        with st.spinner("Classifying..."):
            # Resize and preprocess
            target_size = (224, 224)
            img = image.resize(target_size)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            # Make prediction
            prediction = model.predict(img_array, verbose=0)
            confidence = prediction[0][0]
            
            # Display results
            col1, col2 = st.columns(2)
            
            # Determine label and confidence
            if confidence > 0.5:
                label = "Dog"
                dog_percentage = confidence * 100
                cat_percentage = (1 - confidence) * 100
            else:
                label = "Cat"
                dog_percentage = confidence * 100
                cat_percentage = (1 - confidence) * 100
            
            # Display prediction and confidence
            col1.metric("Prediction", label)
            col2.metric("Confidence", f"{max(cat_percentage, dog_percentage):.2f}%")
            
            # Display confidence bars
            st.write("### Confidence Scores")
            st.progress(cat_percentage / 100)
            st.write(f"**Cat:** {cat_percentage:.2f}%")
            
            st.progress(dog_percentage / 100)
            st.write(f"**Dog:** {dog_percentage:.2f}%")

if __name__ == "__main__":
    main()