"""
Tilapia Disease Detector - Streamlit app for a YOLO26 segmentation model.
Run locally:  streamlit run app.py

Put your trained weights at  model/best.pt  (or change MODEL_PATH below).
Sections marked  >>> EDIT <<<  are the ones you will most likely adjust.
"""

import os
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# =============================================================================
# >>> EDIT <<< 1. CONFIG
# =============================================================================
APP_TITLE = " Nilelytics Tilapia Disease Detector"
MODEL_PATH = "model/best.pt"     # weights from  runs/.../weights/best.pt
IMG_SIZE = 800                   # same as IMG_SIZE used in training
BASE_CONF = 0.25                 # low base threshold; per-class filters applied after

# Per-class minimum confidence (taken from your notebook's inference cell).
# Classes not listed here use DEFAULT_MIN_CONF.
CLASS_MIN_CONF = {
    "Streptococcosis": 0.50,
    "Columnaris": 0.40,
    "Normal": 0.35,
}
DEFAULT_MIN_CONF = 0.35

# Name of the "no disease" class in your dataset
HEALTHY_CLASS = "Normal"

# Optional info shown for each detected class. Keys must match your class names
# (exactly as they appear in data.yaml / model.names).
DISEASE_INFO = {
    "Normal": {
        "description": "No visible signs of disease detected.",
        "action": "Continue regular monitoring and good water quality management.",
    },
    "Columnaris": {
        "description": "Bacterial disease; often shows as pale/white patches, fin erosion and gill damage.",
        "action": "Isolate affected fish, check water quality, and consult a fish health specialist.",
    },
    "Streptococcosis": {
        "description": "Bacterial infection; commonly causes pop-eye, erratic swimming and darkened skin.",
        "action": "Isolate affected fish and consult a veterinarian or aquaculture specialist.",
    },
}


# =============================================================================
# 2. MODEL LOADING (runs once, cached)
# =============================================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)


# =============================================================================
# 3. INFERENCE
# =============================================================================
def run_inference(model, image: Image.Image, base_conf: float, imgsz: int):
    """Returns (annotated RGB image, list of detection dicts)."""
    # Ultralytics treats numpy arrays as BGR, so convert from PIL RGB.
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    result = model.predict(source=bgr, imgsz=imgsz, conf=base_conf, verbose=False)[0]

    keep, detections = [], []
    h, w = result.orig_shape
    for i, box in enumerate(result.boxes):
        name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if conf < CLASS_MIN_CONF.get(name, DEFAULT_MIN_CONF):
            continue

        area_pct = None
        if result.masks is not None:
            poly = result.masks.xy[i]
            if len(poly) >= 3:
                area_pct = 100 * cv2.contourArea(poly.astype(np.float32)) / (h * w)

        keep.append(i)
        detections.append({"Class": name, "Confidence": conf, "Mask area (% of image)": area_pct})

    if keep:
        plotted = result[keep].plot(conf=True, boxes=True, masks=True)
    else:
        plotted = result.orig_img  # nothing passed the filters
    return cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), detections


# =============================================================================
# 4. UI
# =============================================================================
st.set_page_config(page_title="Nilelytics Tilapia Disease Detector", layout="centered")
st.title(APP_TITLE)
st.caption("Upload a clear, well-lit photo of a tilapia to check for signs of disease.")

model = load_model()

with st.sidebar:
    st.header("About")
    st.write(
        "This tool uses an AI segmentation model to screen tilapia images for disease. "
        "It is a decision-support aid, **not** a substitute for a fish health professional."
    )
    if model is not None:
        st.write("**Classes the model detects:**")
        for n in model.names.values():
            st.write(f"- {n}")
    with st.expander("Advanced settings"):
        base_conf = st.slider("Base confidence", 0.05, 0.90, BASE_CONF, 0.05,
                              help="Detections below this are discarded. Per-class filters then apply on top.")
        imgsz = st.select_slider("Inference image size", options=[640, 800, 960], value=IMG_SIZE,
                                 help="Best results when it matches the training size.")

if model is None:
    st.error(
        f"Model file not found at `{MODEL_PATH}`. Copy your trained `best.pt` there "
        "(e.g. from `runs/tilapia-segmentation/<run>/weights/best.pt`) and reload."
    )
    st.stop()

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "jfif", "webp"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption="Input image", use_container_width=True)

    if st.button("ANALYZE", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            annotated, dets = run_inference(model, image, base_conf, imgsz)

        st.subheader("Result")
        st.image(annotated, caption="Detections", use_container_width=True)

        if not dets:
            st.info(
                "No confident detections. The fish may not be clearly visible — "
                "try a closer, sharper, better-lit photo."
            )
        else:
            diseased = [d for d in dets if d["Class"] != HEALTHY_CLASS]
            if not diseased:
                st.success("**No disease detected** — the fish appears normal.")
            else:
                counts = Counter(d["Class"] for d in diseased)
                top = max(diseased, key=lambda d: d["Confidence"])
                st.error(
                    f"**Possible disease detected:** {', '.join(f'{k} ({v})' for k, v in counts.items())}"
                )

                for cls in counts:
                    info = DISEASE_INFO.get(cls)
                    if info:
                        with st.expander(f"About {cls}", expanded=(cls == top["Class"])):
                            st.markdown(f"**Description:** {info['description']}")
                            st.markdown(f"**Suggested action:** {info['action']}")

            df = pd.DataFrame(dets).sort_values("Confidence", ascending=False).reset_index(drop=True)
            df["Confidence"] = (df["Confidence"] * 100).round(1).astype(str) + "%"
            df["Mask area (% of image)"] = df["Mask area (% of image)"].round(1)
            st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.caption("For screening purposes only. Consult a qualified aquaculture veterinarian for diagnosis and treatment.")
