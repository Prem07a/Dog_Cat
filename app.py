import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

class_names = ["Cat", "Dog"]

model_path = './model_train_7'
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
    st.write("My name is Prem Gaikwad, and I am currently pursuing my Bachelor of Engineering degree in Electronics and Telecommunication Engineering (EnTE) from Pune Institute of Computer Technology (PICT), affiliated to Pune University. I am expected to graduate in June 2025.")

    st.header("Education Background")
    st.write("1. Pune Institute of Computer Technology")
    st.write("   - Pune, Maharashtra")
    st.write("   - Bachelor of Engineering (EnTE)")
    st.write("   - Expected - June 2025")

    st.write("2. The Vatsalya School, Pune (CBSE Board)")
    st.write("   - Pune, Maharashtra")
    st.write("   - 12th Grade")
    st.write("   - Year of Completion: 2021")
    st.write("   - Percentage: 82%")

    st.write("3. City International School, Pune (CBSE Board)")
    st.write("   - Pune, Maharashtra")
    st.write("   - 10th Grade")
    st.write("   - Year of Completion: 2019")
    st.write("   - Percentage: 92.2%")

    st.header("Projects")
    st.markdown("1. AI Snake Game\n   - Feb 2023 - Feb 2023\n   - Code: [GitHub Repository](https://github.com/Prem07a/AI_Snake)\n   - Description: I created an AI model based on the concept of Reinforcement Learning. The AI trained itself to play the snake game and learned from its mistakes.")
    
    st.markdown("2. Dog Breed Classification Using PyTorch\n   - Dec 2022 - Jan 2023\n   - Code: [GitHub Repository](https://github.com/Prem07a/Dog-breed-classification)\n   - Description: I created a model that can predict the breed of a dog based on its image. The model was built using PyTorch and transfer learning.")
    
    st.markdown("3. Bulldozer Price Prediction using Machine Learning\n   - Oct 2022 - Oct 2022\n   - Code: [Kaggle Notebook](https://www.kaggle.com/code/premgaikwad07/time-series-bulldozer-price-prediction)\n   - Description: I developed an ML model to predict the price of bulldozers based on various factors. The model was built using Python and the Scikit-Learn library.")
    
    st.markdown("4. Heart Disease Classification using Machine Learning\n   - Oct 2022 - Oct 2022\n   - Code: [Kaggle Notebook](https://www.kaggle.com/code/premgaikwad07/classification-modelling-heart-disease)\n   - Description: I created an ML model to classify whether a person has heart disease or not based on certain factors. The model was trained on historical data and built using Python and the Scikit-Learn library.")

    st.header("Skills")
    st.markdown("- Programming Languages: Python, C, C++")
    st.markdown("- Machine Learning: Regression, Classification, Decision Tree, Linear Models, Ensemble Methods")
    st.markdown("- Deep Learning: Artificial Neural Networks (ANN), Recurrent Neural Networks (RNN), Convolutional Neural Networks (CNN)")
    st.markdown("- Libraries: NumPy, Pandas, Matplotlib, Sci-kit Learn, PyTorch, TensorFlow")

    st.header("Future Goal")
    st.write("To become an AI/ML engineer")


    
    st.write("Thank you for visiting my page and learning more about me!")

  
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

    st.header("Importing Libraries")
    st.code("""
    import os
    import zipfile
    import shutil
    import tensorflow as tf
    from tensorflow.keras import models, layers
    """)

    st.header("Data Preprocessing")
    st.subheader("Setting up Environment")
    st.code("""
    os.environ['KAGGLE_CONFIG_DIR'] = '/content'
    from google.colab import drive
    drive.mount('/content/drive')
    """)

    st.subheader("Downloading the Dataset")
    st.code("""
    !kaggle competitions download -c dogs-vs-cats -p /content/drive/MyDrive/Dog_Cat
    """)

    st.subheader("Extracting the Dataset")
    st.code("""
    dogs_vs_cats_zip_path = '/content/drive/MyDrive/Dog_Cat/dogs-vs-cats.zip'
    output_folder_path = '/content/drive/MyDrive/Dog_Cat'

    with zipfile.ZipFile(dogs_vs_cats_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder_path)

    train_zip_path = os.path.join(output_folder_path, 'train.zip')

    with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder_path)
    """)

    st.header("Image Data Organization")
    st.code("""
    train_folder = '/content/drive/MyDrive/Dog_Cat/train'
    cat_folder = '/content/drive/MyDrive/Dog_Cat/train1/cat'
    dog_folder = '/content/drive/MyDrive/Dog_Cat/train1/dog'

    os.makedirs(cat_folder, exist_ok=True)
    os.makedirs(dog_folder, exist_ok=True)

    for filename in os.listdir(train_folder):
        src_path = os.path.join(train_folder, filename)

        if os.path.isdir(src_path):
            continue

        label = filename.split('.')[0]

        if label == 'cat':
            dest_folder = cat_folder
        elif label == 'dog':
            dest_folder = dog_folder
        else:
            continue

        shutil.move(src_path, dest_folder)
    """)

    st.header("Model Training")
    st.subheader("Loading and Preprocessing the Dataset")
    st.code("""
    IMAGE_SIZE = 80
    BATCH_SIZE = 32
    CHANNELS = 3
    EPOCHS = 50

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "/content/drive/MyDrive/Dog_Cat/train",
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )
    """)

    st.subheader("Train-Validation-Test Split")
    st.code("""
    def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        ds_size = len(dataset)
        if shuffle:
            ds = ds.shuffle(shuffle_size)

        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)

        print(f'TRAIN LENGTH: {len(train_ds)} | VALIDATION LENGTH: {len(val_ds)} | TEST LENGTH: {len(test_ds)} ')
        return train_ds, val_ds, test_ds

    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
    """)

    st.subheader("Data Preprocessing")
    st.code("""
    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.experimental.preprocessing.Rescaling(1.0 / 255)
    ])

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    """)

    st.subheader("Model Architecture")
    st.code("""
    INPUT_SHAPE = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    n_classes = len(dataset.class_names)

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
    """)

    st.subheader("Model Training")
    st.code("""
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=val_ds
    )
    """)

    st.subheader("Saving the Model")
    st.code("""
    model_version = "model_train_4"
    model.save(f'/content/drive/MyDrive/Dog_Cat/{model_version}')
    """)
    
if __name__ == "__main__":
    main()

