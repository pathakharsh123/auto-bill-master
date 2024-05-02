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
        if obj['name'].lower() in item_prices:
            total_bill += item_prices[obj['name'].lower()]
    return total_bill


def detect_objects(results):
    print("ans1", type(results.boxes))
    print("ans2", dir(results.boxes))
    detected_objects = []
    # Check if the results have 'boxes' attribute
    if hasattr(results, 'boxes') and results.boxes:
        for box in results.boxes:
            # Assuming box structure as [x1, y1, x2, y2, confidence, class_id]
            x1, y1, x2, y2 = box.xyxy  # Example, adjust if the actual data structure differs
            class_id = box.class_id  # Example, adjust if needed
            conf = box.confidence  # Example, adjust if needed

            obj_name = results.names[class_id]
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            detected_objects.append(
                {'name': obj_name, 'bbox': bbox, 'conf': conf})

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


if __name__ == "__main__":
    app()
