import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="🧬 Mole Checker",
    layout="centered"
)

st.title("🧬 Mole Checker – Hackathon Demo")

# --------------------------------------------------
# Section 1: Load model
# --------------------------------------------------
MODEL_PATH = "mobilenet_v2_224.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if os.path.exists(MODEL_PATH):
    model = load_model()
    st.success("✅ Model loaded successfully (local version)")
else:
    model = None
    st.warning("⚠️ Local model not found. Demo predictions will be random.")

# --------------------------------------------------
# Section 2: Example images
# --------------------------------------------------
st.markdown("### Example Moles")

col1, col2 = st.columns(2)

with col1:
    st.image("example_benign.jpg", caption="Benign Example", width=200)

with col2:
    st.image("example_malignant.jpg", caption="Malignant Example", width=200)

# --------------------------------------------------
# Section 3: Upload mole image
# --------------------------------------------------
st.markdown("### Upload a Mole Image")

uploaded_file = st.file_uploader(
    "Upload a mole image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    # --------------------------------------------------
    # Preprocess image (MobileNetV2)
    # --------------------------------------------------
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    if model is not None:
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)

        # Assumption: 0 = Benign, 1 = Malignant
        result = "Benign" if predicted_class == 0 else "Malignant"
    else:
        # Demo fallback
        predicted_class = np.random.randint(0, 2)
        confidence = np.random.randint(50, 100)
        result = "Benign" if predicted_class == 0 else "Malignant"

    # --------------------------------------------------
    # Display result
    # --------------------------------------------------
    st.markdown(f"### 🩺 Prediction: *{result}*")
    st.progress(int(confidence))
    st.markdown(f"*Confidence:* {confidence:.2f}%")

    st.warning(
        "⚠️ This is a demo application. Always consult a qualified dermatologist for medical advice."
    )

    # --------------------------------------------------
    # Section 4: Save history
    # --------------------------------------------------
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "image": image.copy(),
        "result": result,
        "confidence": confidence
    })

# --------------------------------------------------
# Section 5: Show history
# --------------------------------------------------
if "history" in st.session_state and len(st.session_state.history) > 0:
    st.markdown("### Upload History (Last 5)")

    for idx, entry in enumerate(st.session_state.history[-5:][::-1], 1):
        st.write(
            f"{idx}. Prediction: *{entry['result']}* | "
            f"Confidence: {entry['confidence']:.2f}%"
        )
        st.image(entry["image"], width=120)
