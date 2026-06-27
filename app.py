import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE = Path(__file__).resolve().parent
with open(BASE / "model_metadata.json") as file:
    config = json.load(file)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(BASE / "best_deployment_model.keras")

model = load_model()
st.title("E-commerce Product Categorizer")
file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    array = np.array(image.resize(tuple(config["image_size"])), dtype=np.float32)
    if config["input_scale_mode"] == "zero_one":
        array = array / 255.0
    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    top = np.argsort(probabilities)[-3:][::-1]
    st.success(f"Prediction: {config['class_names'][int(top[0])].replace('_', ' ').title()} ({probabilities[int(top[0])]:.2%})")
    st.subheader("Top 3 predictions")
    for index in top:
        st.write(f"- {config['class_names'][int(index)].replace('_', ' ').title()}: {probabilities[int(index)]:.2%}")
