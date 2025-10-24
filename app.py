import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------
# App configuration
# --------------------------
st.set_page_config(page_title="🌿 Moringa Leaf Disease Classifier", layout="centered")

st.title("🌿 Moringa Leaf Disease Classifier")
st.write("Upload a Moringa leaf image, and the app will predict the disease class.")

# --------------------------
# Load model (cached)
# --------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("moringa_effnet_final.keras", compile=False)
    return model

model = load_model()

# --------------------------
# Upload image
# --------------------------
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    # --------------------------
    # Preprocess image
    # --------------------------
    img_array = np.array(pil_img, dtype="float32")

    # Ensure 3 channels
    if img_array.ndim == 2:  # grayscale
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[-1] == 1:  # single channel
        img_array = np.concatenate([img_array]*3, axis=-1)

    # Resize to model input size (EfficientNet typically uses 224x224)
    img_resized = tf.image.resize(img_array, (224, 224))

    # Preprocess using EfficientNet's normalization
    img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(
        np.expand_dims(img_resized, axis=0)
    )

    # --------------------------
    # Predict
    # --------------------------
    preds = model.predict(img_preprocessed)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(tf.nn.softmax(preds)) * 100

    # --------------------------
    # Display results
    # --------------------------
    st.success(f"**Predicted Class:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Optional detailed output
    st.write("Raw Model Output:", preds)
else:
    st.info("👆 Please upload an image to begin.")
