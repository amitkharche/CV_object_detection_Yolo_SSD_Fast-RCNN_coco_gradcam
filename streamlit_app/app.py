import streamlit as st
import torch
from PIL import Image
import numpy as np
from src.models.yolov3 import get_yolov3
from src.models.ssd import get_ssd
from src.models.fast_rcnn import get_fast_rcnn

def load_model(model_type):
    if model_type == 'YOLOv3':
        return get_yolov3()
    elif model_type == 'SSD':
        return get_ssd()
    elif model_type == 'Fast R-CNN':
        return get_fast_rcnn()

st.title("📦 Object Detection Demo")
model_type = st.selectbox("Select Model", ['YOLOv3', 'SSD', 'Fast R-CNN'])
show_gradcam = st.checkbox("Show Grad-CAM Overlay")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    model = load_model(model_type)
    st.write(f"Model Loaded: {model_type} (dummy output shown)")
    st.success("Run inference & Grad-CAM here.")
