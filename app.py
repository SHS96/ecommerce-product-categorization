from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_deployment_model.keras"
CONFIG_PATH = BASE_DIR / "model_metadata.json"

st.set_page_config(
    page_title="E-commerce Product Categorizer",
    page_icon="🛍️",
    layout="centered",
)


@st.cache_resource
def load_model_and_config():
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    model = tf.keras.models.load_model(MODEL_PATH)

    return model, config


def prepare_image(image, config):
    resized_image = image.convert("RGB").resize(
        tuple(config["image_size"])
    )

    image_array = np.array(
        resized_image,
        dtype=np.float32,
    )

    if config.get("input_scale_mode") == "zero_one":
        image_array = image_array / 255.0

    return np.expand_dims(image_array, axis=0)


def make_gradcam_heatmap(model, input_batch, predicted_index):
    backbone = model.get_layer("convnext_tiny")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[backbone.output, model.output],
    )

    with tf.GradientTape() as tape:
        convolution_output, predictions = grad_model(
            input_batch,
            training=False,
        )

        selected_class_score = predictions[:, predicted_index]

    gradients = tape.gradient(
        selected_class_score,
        convolution_output,
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    convolution_output = convolution_output[0]

    heatmap = tf.tensordot(
        convolution_output,
        pooled_gradients,
        axes=([2], [0]),
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


def create_gradcam_overlay(original_image, heatmap):
    heatmap_image = Image.fromarray(
        np.uint8(255 * heatmap)
    ).resize(original_image.size)

    colored_heatmap = ImageOps.colorize(
        heatmap_image,
        black="black",
        white="red",
    )

    original_array = np.array(
        original_image.convert("RGB"),
        dtype=np.float32,
    )

    heatmap_array = np.array(
        colored_heatmap.convert("RGB"),
        dtype=np.float32,
    )

    overlay_array = np.uint8(
        np.clip(
            0.60 * original_array
            + 0.40 * heatmap_array,
            0,
            255,
        )
    )

    return Image.fromarray(overlay_array)


model, config = load_model_and_config()

st.title("E-commerce Product Categorizer")

st.write(
    "Upload a product image to receive a predicted category, "
    "confidence score, top-three predictions, and a Grad-CAM explanation."
)

uploaded_file = st.file_uploader(
    "Upload a product image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file).convert("RGB")

    st.image(
        uploaded_image,
        caption="Uploaded image",
        use_container_width=True,
    )

    input_batch = prepare_image(uploaded_image, config)

    probabilities = model.predict(
        input_batch,
        verbose=0,
    )[0]

    top_indices = np.argsort(probabilities)[-3:][::-1]

    predicted_index = int(top_indices[0])
    predicted_class = config["class_names"][predicted_index]
    confidence = float(probabilities[predicted_index])

    st.subheader("Prediction")

    st.success(
        f"Prediction: "
        f"{predicted_class.replace('_', ' ').title()} "
        f"({confidence:.2%})"
    )

    st.subheader("Top 3 Predictions")

    for index in top_indices:
        st.write(
            f"- {config['class_names'][int(index)].replace('_', ' ').title()}: "
            f"{float(probabilities[int(index)]):.2%}"
        )

    st.subheader("Grad-CAM Explanation")

    try:
        heatmap = make_gradcam_heatmap(
            model,
            input_batch,
            predicted_index,
        )

        overlay_image = create_gradcam_overlay(
            uploaded_image,
            heatmap,
        )

        st.image(
            overlay_image,
            caption=(
                "Grad-CAM overlay. Brighter red regions indicate "
                "image areas that contributed more strongly to the "
                "predicted category."
            ),
            use_container_width=True,
        )

        st.info(
            "Grad-CAM is used to visualise which image regions "
            "most influenced the final ConvNeXtTiny prediction."
        )

    except Exception as error:
        st.warning(
            "The product prediction succeeded, but the Grad-CAM "
            "visualisation could not be generated."
        )

        st.code(str(error))
