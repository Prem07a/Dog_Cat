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

def prediction_page():
    st.title("Cat and Dog Classifier")
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if file is not None:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            predicted_class, confidence = predict(image)
            st.write(f"Predicted class: {predicted_class}")
            st.write(f"Confidence: {confidence}%")

def code_page():
    st.title("Code and Explanations")
    st.write("Place your code and explanations here.")

def about_me_page():
    st.title("About Me")
    st.write("My name is Prem Gaikwad, and I am currently pursuing my BE degree in Electronics and Telecommunication Engineering from PICT (Pune Institute of Computer Technology). I have a passion for machine learning and data science, and I love expanding my knowledge by working on projects. This cat and dog classification project is one of my endeavors to learn and apply convolutional neural networks (CNNs) in image classification. I enjoy understanding concepts through practical implementation and believe in the power of data science and analytics to derive insights from data. Feel free to explore this project and learn alongside me!")
    
def main():
    pages = {
        "Image Prediction": prediction_page,
        "Code and Explanations": code_page,
        "About Me": about_me_page
    }

    st.sidebar.title("Navigation")
    page_selection = st.sidebar.radio("Go to", list(pages.keys()))

    current_page = pages[page_selection]
    current_page()

def code_page():
    st.title("Code and Explanations")

    st.header("Data Preprocessing")
    st.code("""
    import os
    os.environ['KAGGLE_CONFIG_DIR'] = '/content'
    from google.colab import drive
    drive.mount('/content/drive')

    !kaggle competitions download -c dogs-vs-cats -p /content/drive/MyDrive/Dog_Cat
    import zipfile
    import os

    # Define the paths
    dogs_vs_cats_zip_path = '/content/drive/MyDrive/Dog_Cat/dogs-vs-cats.zip'
    output_folder_path = '/content/drive/MyDrive/Dog_Cat'

    # Extract the dogs-vs-cats.zip file
    with zipfile.ZipFile(dogs_vs_cats_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder_path)

    # Define the path to the train.zip file
    train_zip_path = os.path.join(output_folder_path, 'train.zip')

    # Extract the train.zip file
    with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder_path)
    """)
    st.write("In this code block, we first set up the necessary environment and mount Google Drive. Then, we download the dataset from the Kaggle competition and specify the paths for the zip files. Next, we extract the 'dogs-vs-cats.zip' file and the 'train.zip' file, which contains the training images. This preprocessing step is required to obtain the dataset for training our cat and dog classifier.")

    st.header("Image Data Organization")
    st.code("""
    import os
    import shutil

    # Path to the 'train' folder in Google Drive
    train_folder = '/content/drive/MyDrive/Dog_Cat/train'

    # Destination folders for cat and dog images
    cat_folder = '/content/drive/MyDrive/Dog_Cat/train1/cat'
    dog_folder = '/content/drive/MyDrive/Dog_Cat/train1/dog'

    # Create the cat and dog folders if they don't exist
    os.makedirs(cat_folder, exist_ok=True)
    os.makedirs(dog_folder, exist_ok=True)

    # Iterate over the files in the train folder
    for filename in os.listdir(train_folder):
        # Construct the source and destination paths
        src_path = os.path.join(train_folder, filename)

        # Skip directories
        if os.path.isdir(src_path):
            continue

        # Extract the class label from the filename
        label = filename.split('.')[0]

        # Define the destination folder based on the class label
        if label == 'cat':
            dest_folder = cat_folder
        elif label == 'dog':
            dest_folder = dog_folder
        else:
            continue  # Skip files with unrecognized labels

        # Move the image to the respective cat or dog folder
        shutil.move(src_path, dest_folder)
    """)
    st.write("In this code block, we organize the training images into separate cat and dog folders. We iterate over the files in the 'train' folder and move each image to the corresponding cat or dog folder based on its label. This step is important to create the necessary directory structure for the dataset.")

    st.header("Model Training")
    st.code("""
    import tensorflow as tf
    from tensorflow.keras import models, layers
    import matplotlib.pyplot as plt
    import numpy as np

    IMAGE_SIZE = 80
    BATCH_SIZE = 32
    CHANNELS = 3
    EPOCHS = 50

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "/content/drive/MyDrive/Dog_Cat/train",
        shuffle=True,
        image_size = (IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE
    )

    class_names = dataset.class_names

    # Train test val split function
    def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        ds_size = len(dataset)
        if shuffle:
            ds = ds.shuffle(shuffle_size)

        train_size = int(train_split*ds_size)
        val_size = int(val_split*ds_size)

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)

        print(f'TRAIN LENGTH: {len(train_ds)} | VALIDATION LENGTH: {len(val_ds)} | TEST LENGTH: {len(test_ds)} ')
        return train_ds, val_ds, test_ds

    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

    # Data preprocessing
    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.experimental.preprocessing.Rescaling(1.0/255)
    ])

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    INPUT_SHAPE = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    n_classes = len(class_names)

    model = models.Sequential([
        resize_and_rescale,
        data_augmentation,

        layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.build(INPUT_SHAPE)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=val_ds
    )

    model_version = "model_train_4"
    model.save(f'/content/drive/MyDrive/Dog_Cat/{model_version}')
    """)
    st.write("In this code block, we perform the model training using TensorFlow and Keras. We define the model architecture with convolutional and dense layers. We also define the data preprocessing steps, including resizing, rescaling, and data augmentation. The training dataset is split into training, validation, and test sets. The model is trained using the training set and evaluated on the validation set. Finally, the trained model is saved for future use.")

    st.write("Feel free to explore the code further and modify it to suit your needs!")

if __name__ == "__main__":
    main()

