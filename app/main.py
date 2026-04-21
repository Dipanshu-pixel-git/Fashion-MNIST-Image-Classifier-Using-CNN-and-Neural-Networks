import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/Fashion_MNIST_CNN_Image_Classifier.h5"
# Load the pretrained model
model = tf.keras.models.load_model(model_path)

# Define Class labels for fashion MNIST dataset
labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
          "Shirt", "Sneaker", "Bag", "Ankleboot"]

# Function to preprocess the Uploaded Image

def preprocess_img(image):
  img = Image.open(image)
  img = img.resize((28,28))
  img = img.convert('L') #Convert to grayscale
  img_array = np.array(img) / 255.0
  img_array = img_array.reshape((1,28,28,1))
  return img_array

# Streamlit Application 
st.title('Fashion Item Classifier')

uploaded_img = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_img is not None:
   image = Image.open(uploaded_img)
   col1, col2 = st.columns(2)

   with col1: 
    resize_img = image.resize((100,100))
    st.image(resize_img)

   with col2:
     if st.button("Classify"):
       #Preprocess the Uploaded Image 
       img_array = preprocess_img(uploaded_img)

       #Make A Prediction Using Pre-trained Model
       result= model.predict(img_array)
       #st.write(str(result))
       predicted_class = np.argmax(result)
       prediction = labels[predicted_class]

       st.success(f'Prediction: {prediction}')


