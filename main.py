import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
from ultralytics import YOLO  # Import the YOLO class


# Define the prices for each item that can be detected.
item_prices = {
    "kitkat": 30,
    "hershey": 40,
    "maggie": 10,
    "cheetos": 50,
    "fanta": 20
}


def calculate_bill(detected_objects):
    total_bill = 0
    for obj in detected_objects:
        obj_lower = obj.lower()
        if obj_lower in item_prices:
            total_bill += item_prices[obj_lower]
    return total_bill

# Function to detect objects using YOLO


def detect_objects(image):
    model = YOLO('best 80.pt')
    results = model(image)  # Perform inference
    detected_objects = []
    for *xyxy, conf, cls in results.xyxy[0]:  # Extract results
        obj_name = model.names[int(cls)]  # Get the object class name
        bbox = [int(x) for x in xyxy]  # Convert bbox to int
        detected_objects.append(
            {'name': obj_name, 'bbox': bbox, 'conf': conf.item()})
    return detected_objects


def capture_image():
    cap = cv2.VideoCapture(0)  # Use the first camera it finds
    ret, frame = cap.read()  # Read one frame from the camera
    cap.release()  # Release the camera

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return pil_image
    else:
        st.error(
            "Failed to capture image from camera. Make sure your webcam is enabled.")
        return None


def app():
    st.header('Auto Billing System')
    st.subheader('Automate your checkout, simplify your life.')

    # Button for capturing the image
    if st.button('Capture Image from Camera'):
        with st.spinner('Capturing image...'):
            image = capture_image()
            if image is not None:
                st.image(image, caption='Captured Image',
                         use_column_width=True)

                # Convert PIL Image to a numpy array
                img_array = np.array(image)

                # Detect objects in the image
                detected_objects = detect_objects(img_array)
                if detected_objects:
                    st.subheader("Detected Objects and their bounding boxes:")
                    for obj in detected_objects:
                        st.write(
                            f"{obj['name']} ({obj['conf']:.2f}): {obj['bbox']}")

                    total_bill = calculate_bill(detected_objects)
                    st.subheader("Total Bill:")
                    st.write("₹", total_bill)
                else:
                    st.write("No objects detected.")
            else:
                st.write("Failed to capture image.")


# Fix the typo in if _name_ check
if __name__ == "__main__":
    app()
