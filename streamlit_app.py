import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load the saved model
model = tf.keras.models.load_model('hecr.keras')

# Define a dictionary for label mapping
label_map = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',
    11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',
    21:'V',22:'W',23:'X',24:'Y',25:'Z'
}

# Streamlit app title and instructions
st.title("Handwritten Alphabet Recognition")
st.write("Upload an image of a handwritten alphabet to see the prediction.")

# File uploader for input image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image)
    image_array = cv2.threshold(image_array, 30, 200, cv2.THRESH_BINARY)[1]  # Apply thresholding
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for the model
    image_array = image_array / 255.0  # Normalize

    # Prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = label_map[predicted_class]
    
    # Display the prediction
    st.write(f"Prediction: {predicted_label}")