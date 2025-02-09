import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

# Load the trained model
model = tf.keras.models.load_model("waste_classifier.h5")

st.title("♻ Waste Classification App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((64, 64))  

    img_array = np.array(image) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    
    # Use threshold for binary classification
    predicted_class = "Recyclable" if predictions[0][0] > 0.5 else "Organic"

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"### 🏷 Predicted Class: {predicted_class}")
