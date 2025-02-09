import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import streamlit as st

# Load the trained model
model = keras.models.load_model("waste_classifier.h5")

# Ensure class labels match the number of model output classes
class_labels = ["Organic", "Recyclable"]  # Only 2 classes!

st.title("♻ Waste Classification App")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image and preprocess
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((64, 64))  # Resize to match model input shape

    # Convert image to NumPy array
    img_array = np.array(image) / 255.0  # Normalize (0-1)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 64, 64, 3)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Ensure the predicted index is valid
    if predicted_class >= len(class_labels):  
        st.write("⚠ Error: Model returned an invalid class index.")
    else:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"### 🏷 Predicted Class: {class_labels[predicted_class]}")
