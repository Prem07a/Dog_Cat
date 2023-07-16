import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

CLASS_NAMES = ["Cat", "Dog"]

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("saved_model\model_1") 
    return model

model = load_model()

st.write("# Cat and Dog Classification")

file = st.file_uploader("Please Upload an Image of a Cat or Dog", type=["jpg", "png"])

def import_and_predict(image, model):
    size = (256, 256)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    model_output = model.predict(img_reshape)
    return model_output

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    model_output = import_and_predict(image, model)
    confidence = np.max(model_output) * 100
    prediction_index = np.argmax(model_output)
    prediction_label = CLASS_NAMES[prediction_index]
    string = f"This image is classified as: {prediction_label} with confidence: {confidence:.2f}%"
    st.success(string)
