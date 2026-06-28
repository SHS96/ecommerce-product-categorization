from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

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
        dtype=np.float32
    )

    if config.get("input_scale_mode") == "zero_one":
        image_array = image_array / 255.0

    return np.expand_dims(image_array, axis=0)


def build_gradcam_model(model):
    """
    Rebuilds the ConvNeXt feature path from a fresh input tensor.
    This avoids the disconnected-graph error from the saved nested model.
    """
    gradcam_input = tf.keras.Input(
        shape=model.input_shape[1:],
        name="gradcam_input"
    )

    augmentation = model.get_layer("convnext_augmentation")
    backbone = model.get_layer("convnext_tiny")

    x = augmentation(gradcam_input, training=False)
    conv_features = backbone(x, training=False)

    backbone_index = model.layers.index(backbone)
    head_output = conv_features

    for layer in model.layers[backbone_index + 1:]:
        if isinstance(
            layer,
            (
                tf.keras.layers.Dropout,
                tf.keras.layers.BatchNormalization,
            ),
        ):
            head_output = layer(head_output, training=False)
        else:
            head_output = layer(head_output)

    return tf.keras.Model(
        inputs=gradcam_input,
        outputs=[conv_features, head_output],
        name="gradcam_model"
    )


@st.cache_resource
def load_gradcam_model():
    model, _ = load_model_and_config()
    return build_gradcam_model(model)


def make_gradcam_heatmap(
    gradcam_model,
    input_batch,
    predicted_index
):
    with tf.GradientTape() as tape:
        conv_features, predictions = gradcam_model(
            input_batch,
            training=False
        )

        selected_score = predictions[:, predicted_index]

    gradients = tape.gradient(selected_score, conv_features)

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_features = conv_features[0]

    heatmap = tf.reduce_sum(
        conv_features * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


def create_gradcam_overlay(original_image, heatmap):
    base_image = original_image.convert("RGBA")

    heatmap_image = Image.fromarray(
        np.uint8(255 * heatmap)
    ).resize(
        base_image.size,
        Image.Resampling.BILINEAR
    )

    heatmap_array = np.array(
        heatmap_image,
        dtype=np.float32
    ) / 255.0

    red_overlay = Image.new(
        "RGBA",
        base_image.size,
        (255, 0, 0, 0)
    )

    alpha_mask = Image.fromarray(
        np.uint8(heatmap_array * 155)
    )

    red_overlay.putalpha(alpha_mask)

    return Image.alpha_composite(
        base_image,
        red_overlay
    ).convert("RGB")


model, config = load_model_and_config()

st.title("E-commerce Product Categorizer")

st.write(
    "Upload a product image to receive a predicted category, "
    "confidence score, top-three predictions, and a Grad-CAM explanation."
)

uploaded_file = st.file_uploader(
    "Upload a product image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file).convert("RGB")

    st.image(
        uploaded_image,
        caption="Uploaded image",
        use_container_width=True
    )

    input_batch = prepare_image(uploaded_image, config)

    probabilities = model.predict(
        input_batch,
        verbose=0
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
        category_name = (
            config["class_names"][int(index)]
            .replace("_", " ")
            .title()
        )

        st.write(
            f"- {category_name}: "
            f"{float(probabilities[int(index)]):.2%}"
        )

    st.subheader("Grad-CAM Explanation")

    try:
        gradcam_model = load_gradcam_model()

        heatmap = make_gradcam_heatmap(
            gradcam_model,
            input_batch,
            predicted_index
        )

        overlay_image = create_gradcam_overlay(
            uploaded_image,
            heatmap
        )

        st.image(
            overlay_image,
            caption=(
                "Grad-CAM overlay. Brighter red regions contributed "
                "more strongly to the predicted category."
            ),
            use_container_width=True
        )

        st.info(
            "Grad-CAM highlights image regions that most influenced "
            "the final ConvNeXtTiny prediction."
        )

    except Exception as error:
        st.warning(
            "The product prediction succeeded, but the Grad-CAM "
            "visualisation could not be generated."
        )
        st.code(str(error))