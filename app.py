import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load model and haarcascade
model = load_model("emotion_detector.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# UI
st.title("Emotion Detector")
st.write("Upload a face image to detect emotion.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image.convert('L'))  # convert to grayscale
    faces = face_cascade.detectMultiScale(img_array, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected.")
    else:
        for (x, y, w, h) in faces:
            roi = img_array[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi / 255.0
            roi = roi.reshape(1, 48, 48, 1)

            prediction = model.predict(roi)
            predicted_emotion = emotions[np.argmax(prediction)]

            st.success(f"Detected Emotion: **{predicted_emotion}**")
