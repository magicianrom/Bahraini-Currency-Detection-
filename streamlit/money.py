
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Load trained YOLO model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "train-7" / "weights" / "best.pt"

model = YOLO(str(MODEL_PATH))

st.title("Bahraini Currency Detection")
st.write("Upload an image of Bahraini currency to detect its denomination.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded image")

    # Predict
    results = model.predict(
        source=image,
        device="cpu"
    )

    # Display image with bounding boxes
    result_image = results[0].plot()[:, :, ::-1]

    st.subheader("Detection Results")
    st.image(result_image)

    # Display detected classes and confidence
    for box in results[0].boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)

        class_name = model.names[class_id]

        st.write(
            f"Detected: {class_name} BHD | Confidence: {confidence:.2%}"
        )

    if len(results[0].boxes) == 0:
        st.write("No currency detected.")
