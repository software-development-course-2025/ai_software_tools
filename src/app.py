# streamlit.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✍️", layout="wide")

# --- 2. LOAD THE TRAINED MODEL ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('mnist_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- 3. APP UI AND LOGIC ---
st.title("✍️ MNIST Handwritten Digit Classifier")
st.markdown("""
This app uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to classify handwritten digits (0-9).
This fulfills the **Bonus Task** of the "Mastering the AI Toolkit" assignment.
""")

# Create two columns for the layout
col1, col2 = st.columns([1, 1])

# --- LEFT COLUMN (IMAGE UPLOAD) ---
with col1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose a digit image (28x28 pixels preferred)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None and model is not None:
        # Open the image
        image = Image.open(uploaded_file)

        # Preprocess the image to match the model's input requirements
        image_gray = image.convert('L') # Convert to grayscale
        image_resized = image_gray.resize((28, 28)) # Resize to 28x28 pixels
        
        # Invert colors (MNIST digits are white on black background)
        image_inverted = Image.eval(image_resized, lambda x: 255 - x)

        # Convert to numpy array and normalize
        image_array = np.array(image_inverted) / 255.0
        
        # Reshape for the model (add batch and channel dimensions)
        image_reshaped = image_array.reshape(1, 28, 28, 1)

# --- RIGHT COLUMN (RESULTS) ---
with col2:
    st.subheader("Analysis Result")
    if uploaded_file is None:
        st.info("Please upload an image to see the classification result.")
    
    if uploaded_file is not None and model is not None:
        # Display the uploaded and processed image
        st.image(image_inverted, caption='Processed Image (28x28)', width=150)
        
        # Make a prediction
        prediction = model.predict(image_reshaped)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Display the result
        st.success(f"The model predicts this digit is a: **{predicted_digit}**")
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
