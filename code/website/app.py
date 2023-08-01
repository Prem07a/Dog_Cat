import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL = tf.keras.models.load_model("./models/model_train_8")
CLASS_NAMES = ['CAT', 'DOG']

def read_file_as_image(data) -> np.ndarray:
    """
    Reads the uploaded file data and converts it into a numpy array (image representation).

    Args:
        data (BytesIO): The raw bytes of the uploaded file.

    Returns:
        np.ndarray: The image data as a numpy array.
    """
    image = np.array(Image.open(data))
    return image

def predict(image):
    """
    Performs image classification on the provided image.

    Args:
        image (np.ndarray): The image data as a numpy array.

    Returns:
        tuple: A tuple containing the predicted class and confidence.
    """
    # Resize the image to (80, 80) as expected by the model
    image = tf.image.resize(image, (80, 80))
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = ((np.max(predictions[0]) * 100 * 100) // 1) / 100

    return predicted_class, confidence

def main():
    """
    Main function to create the Image Classification Web App.

    This function sets up the Streamlit web app and allows users to upload an image.
    When the "Predict" button is pressed, it performs image classification and displays
    the predicted class and confidence score on the web app.
    """
    st.title("Dog & Cat Image Classification")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = read_file_as_image(uploaded_file)

        # Add CSS style to display the image with padding from the left
        st.image(image, caption=f"Uploaded Image", width=250, use_column_width=False, clamp=True)

        # Add a "Predict" button to trigger the prediction
        if st.button("Predict"):
            predicted_class, confidence = predict(image)

            st.write(f"Class: {predicted_class}")
            st.write(f"Confidence: {confidence}%")

if __name__ == "__main__":
    main()
