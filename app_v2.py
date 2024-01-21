from secrets import choice
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp
from PIL import Image
import matplotlib.pyplot as plt
import io
import subprocess
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Dense
import efficientnet.tfkeras as efn
from tensorflow.keras import layers as L
import glob
import os
from PIL import Image
import contextlib
import time

# Function to load and preprocess image
def load_and_preprocess_image(image_file):
    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, width=300)  # Display the uploaded image
        image = image.resize((256, 256))  # Resize the image to (256, 256)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.cast(image, tf.float32) / 255.0
        img_size = [256, 256]
        image = tf.reshape(image, [*img_size, 3])
        image = tf.expand_dims(image, axis=0)
        return image
    return None



# Function to create the skin cancer detection model
def create_skin_cancer_model():
    model = tf.keras.Sequential([
        efn.EfficientNetB2(
            input_shape=(*[256, 256], 3),
            weights='imagenet',
            include_top=False
        ),
        L.GlobalAveragePooling2D(),
        L.Dense(1024, activation='relu'),
        L.Dropout(0.3),
        L.Dense(512, activation='relu'),
        L.Dropout(0.2),
        L.Dense(256, activation='relu'),
        L.Dropout(0.2),
        L.Dense(128, activation='relu'),
        L.Dropout(0.1),
        L.Dense(1, activation='sigmoid')
    ])
    return model

# Function to perform image prediction
def predict_image(model, image):
    pred = model.predict(image)
    pred_per = round(pred[0][0] * 100, 2)
    return pred_per

# Function for the "Image Detection" page
def image_detection_page():
    st.write("The cells that make melanin, the pigment responsible for your skin's color, can grow into melanoma, the most dangerous kind of skin cancer. Melanoma can also develop in your eyes and, very rarely, within your body, including in your throat or nose.")
    st.write("Here you can insert your own skin legion image and have the app predict how likely it is to be a postitive case of melanoma.")
    st.write("You can select one of the models below.")

    files = os.listdir('workspace/clientResults')
    weight_files = [i for i in files if 'round' in i or 'base' in i]

    selected_weight = st.selectbox("Select model to predict", weight_files)
    model = create_skin_cancer_model()

    if selected_weight is not None:
        model.load_weights('./workspace/clientResults/' + selected_weight)
        st.info('Loaded model weights: ' + selected_weight)

    image_file = st.file_uploader("Upload Image", type=["jpg"])

    if image_file is not None:
        image = load_and_preprocess_image(image_file)
        pred_per = predict_image(model, image)

        st.info(f'Melanoma risk is high by %{pred_per}' if pred_per > 50 else f'Melanoma risk is low by %{pred_per}')

# Function for the "Data Exploration" page
def data_exploration_page():
    st.subheader("Data Exploration")
    st.write("Here you can see the specifications of the data")

    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Your dataset")
        st.write(df)

        st.subheader("Visualization of your dataset")

        # Count plot for diagnosis distribution
        st.write("Diagnosis Distribution:")
        fig, ax = plt.subplots()
        sns.countplot(x='diagnosis', data=df, palette='viridis', ax=ax)
        st.pyplot(fig)

        # Age distribution
        st.write("Age Distribution:")
        fig, ax = plt.subplots()
        sns.histplot(df['age'], kde=True, color='skyblue', bins=30, ax=ax)
        st.pyplot(fig)

        # Gender distribution
        st.write("Gender Distribution:")
        fig, ax = plt.subplots()
        sns.countplot(x='sex', data=df, palette='muted', ax=ax)
        st.pyplot(fig)

        # Anatomical site distribution
        st.write("Anatomical Site Distribution:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='location', data=df, palette='pastel', ax=ax, order=df['location'].value_counts().index)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

        # Pair plot for age and diagnosis
        st.write("Pair Plot for Age and Diagnosis:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols[numeric_cols != 'id']  # Exclude 'id' column if present
        fig = sns.pairplot(df, hue='diagnosis', vars=numeric_cols, palette='dark', markers=['o', 's'], height=3)
        st.pyplot(fig)

# Function for the "Training" page
def training_page():
    test_df = st.file_uploader("Choose a file", key='diagnosis')

    # Function to run a command and display real-time progress
    def run_command(args):
        st.info(f"Federated Learning network is now training with 1 server and 2 clients")
        result = subprocess.run(args, capture_output=True, text=True)

        try:
            result.check_returncode()

            # Display training progress and logs
            for line in result.stdout.splitlines():
                if "Epoch" in line:
                    epoch_info = line.split()
                    current_epoch = int(epoch_info[1])
                    total_epochs = int(epoch_info[3])
                    progress_percent = (current_epoch / total_epochs) * 100
                    st.progress(progress_percent)

                # Customize this part based on the actual format of your training logs
                # For example, displaying lines containing "Loss" or "Accuracy"
                if "Loss" in line or "Accuracy" in line:
                    st.write(line)

            st.success("Training completed successfully.")
        except subprocess.CalledProcessError as e:
            st.error(result.stderr)
            raise e

    if st.button("Train"):
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                with contextlib.redirect_stderr(stderr):
                    with st.spinner(f'Training with {test_df.name}...'):
                        run_command(["bash", "run.sh", "-p", f"isicdata/datasets/{test_df.name}"])
        except Exception as e:
            st.write(f"Failure while executing: {e}")

# Main function to run the Streamlit app
def main():
    st.sidebar.image("logo.png", use_column_width=True)
    st.sidebar.title("Skin Cancer Detection through Neural Network on Federated Learning")

    page_names_to_funcs = {
        "Image Detection": image_detection_page,
        "Data Exploration": data_exploration_page,
        "Training": training_page,
    }

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()

if __name__ == "__main__":
    main()
