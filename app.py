import streamlit as st
import numpy as np
import os
from PIL import Image
import joblib
import time

# Streamlit app title and description
st.title("Human Detection")
st.write("Upload an image and see the prediction")

model = joblib.load("/Users/godfather_101/Downloads/human detection dataset/Model")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Check if an image is uploaded
if uploaded_file is not None:
    # Open and display the uploaded image using PIL
    img = Image.open(uploaded_file)
    img = img.resize((128, 128))  # Resize the image

    # Display the image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Convert the image to a numpy array
    img_array = np.array(img)

    if st.button("Predict"):
        # Reshape the image array to match the input shape of the model
        img_array = img_array.reshape((1, 128, 128, 3))

        # Predict using the loaded model
        prediction = model.predict(img_array)

        with st.spinner():
            time.sleep(1)  # Simulate a delay for demonstration purposes
            if prediction > 0.5:
                st.success("Human Detected")
            else:
                st.error("Human Not Detected")
