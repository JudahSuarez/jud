import os
import time
from pathlib import Path
from threading import Lock

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO


st.set_page_config(page_title="Live Object Detection", page_icon="🎥", layout="centered")

MODEL_PATH = "yolov8n.pt"
OUTPUT_DIR = Path("detected_frames")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_CLASSES = {"person", "cell phone", "bottle"}
SAVE_CLASSES = {"person", "bottle"}

state_lock = Lock()
latest_counts = {name: 0 for name in TARGET_CLASSES}
latest_warning = ""
last_save_time = 0.0


@st.cache_resource
def load_model():
    """Load the YOLOv8 model only once."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}. Please place yolov8n.pt in the same folder as enhance.py.")
        st.stop()
    return YOLO(MODEL_PATH)


model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real time.")

confidence_threshold = st.slider(
    "Detection Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

st.info("Detected objects are shown on the video. Frames are saved when a person or bottle is detected.")

counts_placeholder = st.empty()
warning_placeholder = st.empty()


def video_frame_callback(frame):
    """Process each webcam frame with YOLOv8."""
    global latest_counts, latest_warning, last_save_time

    img = frame.to_ndarray(format="bgr24")

    results = model.track(
        img,
        persist=True,
        conf=confidence_threshold,
        verbose=False,
    )

    annotated_frame = results[0].plot()

    counts = {name: 0 for name in TARGET_CLASSES}
    detected_objects = []

    if results[0].boxes is not None and results[0].boxes.cls is not None:
        for cls_id in results[0].boxes.cls:
            class_name = results[0].names[int(cls_id)]
            detected_objects.append(class_name)
            if class_name in counts:
                counts[class_name] += 1

    warning_message = "Cell phone detected!" if "cell phone" in detected_objects else ""


    now = time.time()
    if any(obj in SAVE_CLASSES for obj in detected_objects) and now - last_save_time >= 2:
        frame_name = OUTPUT_DIR / f"frame_{int(now)}.jpg"
        cv2.imwrite(str(frame_name), annotated_frame)
        last_save_time = now

    with state_lock:
        latest_counts = counts
        latest_warning = warning_message

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


ctx = webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
    },
    media_stream_constraints={"video": True, "audio": False},
)

with state_lock:
    counts_placeholder.write(f"Detected Objects: {latest_counts}")
    if latest_warning:
        warning_placeholder.warning(latest_warning)
