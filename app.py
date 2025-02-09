
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import streamlit as st

# Load the trained model
model = keras.models.load_model("waste_classifier.h5")

# Class labels (modify if needed)
class_labels = ["Organic", "Recyclable"]  # Change to actual class names

st.title("♻ Waste Classification App")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")  # Ensure 3 channels

    # Convert image to NumPy array
    img_array = np.array(image) / 255.0  # Normalize (0-1)

    # Expand dimensions (Add batch size)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"### 🏷 Predicted Class: {class_labels[predicted_class]}")
