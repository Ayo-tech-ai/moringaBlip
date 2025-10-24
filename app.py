
# --------------------------- imports ---------------------------
import streamlit as st

# Page config MUST be before any other Streamlit command
st.set_page_config(page_title="Moringa Leaf Disease Detector", layout="wide")

import tensorflow as tf
import numpy as np
from PIL import Image
from tf_explain.core.grad_cam import GradCAM

# --------------------------- custom CSS ------------------------
st.markdown(
    """
    <style>
    .stButton>button {
        background-color:#4CAF50;
        color:white;
        font-weight:bold;
        border:none;
        padding:0.6em 1.2em;
        border-radius:6px;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background-color:#45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------ disease info ------------------------
DISEASE_INFO = {
    "Bacterial Leaf Spot": {
        "name": "Bacterial Leaf Spot",
        "cause": "Bacteria (Xanthomonas / Pseudomonas spp.) infecting leaf tissue",
        "symptoms": (
            "Small, water-soaked specks that enlarge, darken, and may ooze under humid conditions."
        ),
        "management": [
            "Remove and destroy infected leaves.",
            "Avoid overhead watering; keep foliage dry.",
            "Apply copper-based bactericide early in the outbreak.",
        ],
    },
    "Cercospora Leaf Spot": {
        "name": "Cercospora Leaf Spot",
        "cause": "Fungal infection by *Cercospora moringicola*",
        "symptoms": (
            "Circular grey-brown lesions with yellow halos; severe cases cause premature leaf drop."
        ),
        "management": [
            "Collect and burn fallen leaves to reduce spores.",
            "Apply protectant fungicide (e.g., mancozeb) during rainy periods.",
            "Prune overcrowded branches to improve air flow.",
        ],
    },
    "Healthy Leaf": {
        "name": "Healthy Leaf",
        "cause": "No signs of disease or stress detected",
        "symptoms": (
            "Leaf appears green, intact, and vibrant — no visible lesions, discoloration, or abnormalities."
        ),
        "management": [
            "Maintain regular watering and balanced fertilisation.",
            "Continue monitoring for any future changes.",
            "Keep surrounding area clean to reduce disease risk.",
        ],
    },
    "Yellow Leaf": {
        "name": "Nutrient / Water Stress (Yellow Leaf)",
        "cause": "Typically nitrogen or iron deficiency; sometimes over-watering",
        "symptoms": (
            "Uniform yellowing beginning on older leaves; veins may stay green if iron is lacking."
        ),
        "management": [
            "Apply balanced NPK fertiliser or iron chelate.",
            "Check soil drainage; avoid prolonged water-logging.",
            "Mulch to maintain even soil moisture.",
        ],
    },
}

CLASS_NAMES = [
    "Bacterial Leaf Spot",
    "Cercospora Leaf Spot",
    "Healthy Leaf",
    "Yellow Leaf",
]

# ------------------------ load model ------------------------
@st.cache_resource
def load_model():
    m = tf.keras.models.load_model("moringa_effnet_final.keras")
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return m

model = load_model()
backbone = model.get_layer("efficientnetb0")

# ---------------------- grad-cam helper ----------------------
def generate_gradcam(batch, model, class_idx, target_layer="block5a_project_conv"):
    explainer = GradCAM()
    return explainer.explain(
        validation_data=(batch, None),
        model=model,
        class_index=class_idx,
        layer_name=target_layer,
    )

# -------------------------- UI --------------------------
st.title("🌿 Moringa Leaf Disease Detector with Grad-CAM")

uploaded_file = st.file_uploader("Upload a moringa leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Running prediction..."):
            # Preprocess
            arr = tf.keras.applications.efficientnet.preprocess_input(
                np.expand_dims(np.array(pil_img, dtype="float32"), axis=0)
            )

            # Predict with full model
            preds = model(arr, training=False)
            class_idx = int(np.argmax(preds))
            class_name = CLASS_NAMES[class_idx]

            st.success(f"🧠 Predicted class: **{class_name}**")

            # Grad-CAM heatmap
            heatmap = generate_gradcam(arr, backbone, class_idx)
            st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)

            # Disease info
            info = DISEASE_INFO[class_name]
            with st.expander("🩺 Disease Information", expanded=True):
                st.markdown(f"### {info['name']}")
                st.caption(f"**Cause:** {info['cause']}")
                st.markdown(f"**Symptoms:** {info['symptoms']}")
                st.markdown("**Management Tips:**")
                for tip in info["management"]:
                    st.markdown(f"- {tip}")

            # New-image button
            if st.button("🔄 New Image"):
                st.experimental_rerun()
