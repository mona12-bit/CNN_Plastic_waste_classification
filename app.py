import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

# Load the trained model
model = tf.keras.models.load_model("waste_classifier.h5")

st.title("♻ Waste Classification App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Open and preprocess the image
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((64, 64))  # Resize to match model input shape

        # Convert to NumPy array
        img_array = np.array(image) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Debugging print
        print("Image Array Shape:", img_array.shape)

        # Predict
        predictions = model.predict(img_array)  
        print("Raw Model Output:", predictions)  # Debugging

        # Use threshold for binary classification
        threshold = 0.5  # Adjust this if needed
        predicted_class = "Recyclable" if predictions[0][0] > threshold else "Organic"

        # Display the result
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"### 🏷 Predicted Class: {predicted_class}")

    except Exception as e:
        st.error(f"⚠ Error: {str(e)}")  # Show error in UI

else:
    st.write("📂 Please upload an image.")
