import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import streamlit as st

# Load model
model = keras.models.load_model("waste_classifier.h5")

# Define class labels
class_labels = ["Organic", "Recyclable"]  # Only 2 classes

st.title("♻ Waste Classification App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((64, 64))  

    img_array = np.array(image) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    predictions = model.predict(img_array)
    print("Model Raw Predictions:", predictions)  # Debugging

    if model.output_shape[-1] == 1:  
        predicted_class = int(predictions[0][0] > 0.5)  # Binary classification
    else:
        predicted_class = np.argmax(predictions, axis=1)[0]  # Multiclass classification

    if predicted_class >= len(class_labels):  
        st.write("⚠ Error: Model returned an invalid class index.")
    else:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"### 🏷 Predicted Class: {class_labels[predicted_class]}")
