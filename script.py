import cv2
import streamlit as st
import os
from datetime import datetime
import numpy as np
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False



SAVE_DIR = "detected_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

last_called = time.time() 

def detect_faces(color, scale_factor, min_neighbors):
    cap = cv2.VideoCapture(0)

    stframe = st.empty()

    last_called = time.time() 

    while st.session_state.run_detection:
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to read from webcam.")
            break
        #frame = cv2.imread('faces.examples.jpg')

        #print("Image not found!")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor,  minNeighbors=min_neighbors)
    
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            #cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        current_time = time.time()
        if current_time - last_called >= 30:
        #if st.button("Save Image", key="save"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SAVE_DIR, f"face_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            st.success(f"Image saved: {filename}")

            last_called = current_time
        
        time.sleep(1)

        # Add a way to break the loop manually
        #if st.button("Stop Detection", key="stop"):
         #   break

    cap.release()
    cv2.destroyAllWindows()
 

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write(
        "Press the button below to start detecting faces from your webcam\n" \
        "1. Click Start Face Detection to begin.\n"\
        "2. Use sliders to adjust detection sensitivity.\n"\
        "3. Pick a rectangle color to highlight faces.\n"\
        "4. A snapshot will be saved to your device every 30 seconds.\n"\
        "5. Click Stop Detection to stop webcam.\n"\
        )
    
    rect_color = st.color_picker("Pick rectangle color", '#00FF00')
    color_rgb = tuple(int(rect_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    scale_factor = st.slider("Scale Factor", 1.05, 1.5, 1.1, 0.01)
    min_neighbors = st.slider("Min Neighbors", 3, 10, 5, 1)


    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Face Detection"):
            st.session_state.run_detection = True
            detect_faces(color_rgb, scale_factor, min_neighbors)

    with col2:
        if st.button("Stop Detection"):
            st.session_state.run_detection = False


if __name__ == "__main__":
    app()