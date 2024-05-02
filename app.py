import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
from ultralytics import YOLO
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

# This function is a placeholder for the YOLO detection model
# In reality, you should replace this with actual model prediction results


def detect_objects(image):
    model = YOLO('best1.pt')
    results = model.predict(source=image)
    print("ans1", type(results))
    print("ans2", dir(results))
    print("ans3", results.pred)
    detected_objects = []  # List to store detected object names
    for result in results:
        if result.boxes:
            for box in result.boxes:
                class_id = int(box.cls)
                object_name = model.names[class_id]
                # Append object name to the list
                detected_objects.append(object_name)

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

                # Simulate a delay for object detection
                # Simulating time delay as if processing is happening
                time.sleep(2)

                detected_objects = detect_objects(image)
                if detected_objects:
                    st.subheader("Detected Objects:")
                    for obj in detected_objects:
                        st.write(obj)

                    total_bill = calculate_bill(detected_objects)
                    st.subheader("Total Bill:")
                    st.write("₹", total_bill)
                else:
                    st.write("No objects detected.")
            else:
                st.write("Failed to capture image.")


if __name__ == "__main__":
    app()
