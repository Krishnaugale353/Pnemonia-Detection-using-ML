import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Replace 'our_modell.h5' with your actual model path
model = load_model('our_modell.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    image = img_to_array(img)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# Classes for prediction labels (assuming class folder names)
classes = ['NORMAL', 'PNEUMONIA']
def main():
  st.title("Pneumonia Detection App")
  st.text("Upload an X-ray image to predict pneumonia.")

  uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

  if uploaded_file is not None:
    image = preprocess_image(uploaded_file.name)  # Preprocess the image
    prediction = model.predict(image)[0]  # Get prediction probabilities
    predicted_class = classes[np.argmax(prediction)]  # Get the class with highest probability

    st.subheader("Prediction:")
    st.text(predicted_class)

    # Display additional information based on prediction (optional)
    if predicted_class == "PNEUMONIA":
      st.warning("This is a predicted result. Please consult a medical professional for diagnosis and treatment.")
    else:
      st.success("The model predicts no pneumonia in the image. This is not a definitive diagnosis, and it is still recommended to consult a doctor if you have any concerns.")

if __name__ == '__main__':
  main()
