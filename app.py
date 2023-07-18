import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

class_names = ["Cat", "Dog"]

model_path = './model_train_5'
model = tf.keras.models.load_model(model_path)

def predict(image):
    image = image.resize((80, 80))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = int(np.max(prediction[0]) * 100)
    return predicted_class, confidence

def main():
    st.title("Cat and Dog Classifier")
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if file is not None:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            predicted_class, confidence = predict(image)
            st.write(f"Predicted class: {predicted_class}")
            st.write(f"Confidence: {confidence}%")

if __name__ == "__main__":
    main()
