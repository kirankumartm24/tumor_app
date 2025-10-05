import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("model/brain_tumor_model.h5")

IMG_SIZE = (224, 224)

st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI scan to check for the presence of a brain tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(img, caption='Uploaded MRI', use_container_width=True)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    label = "🟥 Tumor Detected" if prediction > 0.5 else "🟩 No Tumor Detected"

    st.subheader(f"Prediction: {label} ")