import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
import tensorflow as tf

model = tf.keras.models.load_model("waste_classifier.h5")  # Load the saved model

# Define class names (Modify based on your dataset)
class_names = ["Organic", "Recyclable"]

st.title("♻️ Waste Classification App")
st.write("Upload an image to classify whether it's **Organic** or **Recyclable**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))  # Resize to match model input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]
    
    # Display the result
    st.success(f"### ✅ Prediction: {predicted_class}")
