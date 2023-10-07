import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import joblib
import time

# Streamlit app title and description
st.title("Human Detection ")
st.write("Upload an image and see the prediction")

model = joblib.load("/Users/godfather_101/Downloads/human detection dataset/Model")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Check if an image is uploaded
if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.read())

    # Read the uploaded image using OpenCV
    img = cv2.imread(temp_image_path)
    img = cv2.resize(img,(128,128))
    img = np.expand_dims(img,axis=0)

    
    if st.button("Predict"):
        prediction = model.predict(img)
        with st.spinner():
            time.sleep(1)
            if prediction > .5:
                st.success("Human Detection")
            else:
                st.success("Human Not Detection")

    # Remove the temporary image file
    os.remove(temp_image_path)




