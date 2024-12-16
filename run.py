import streamlit as st
import numpy as np
from PIL import Image
# from flask import request, jsonify

import keras
model = keras.models.load_model('best_model.keras')

def resize_image(img_file, target_size=(224, 224)):
    return img_file.resize(target_size)

def process(img_file):
    img = resize_image(img_file)        # Resize image to match model input
    img_array = np.array(img)           # Convert image to numpy array
    if img_array.shape[-1] != 3:        # Ensure 3-channel image
        img_array = img_array[..., :3]
    img_array = img_array / 255.0       # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def main():
    st.title("Camera Image AI Processor")

    st.header("Camera Input")
    # img_file_buffer = st.camera_input("Take a picture")
    img_file_buffer = st.file_uploader("Upload a picture")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        processed = process(image)
        res = model.predict(processed)
        st.write(res)
        print(res)

        

if __name__ == "__main__":
    main()
