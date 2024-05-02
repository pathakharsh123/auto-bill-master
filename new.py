import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2

item_prices = {
    "kitkat": 10,
    "hershey": 65,
    "maggie": 20,
    "lays": 10,
    "closeup": 20,
    "maaza": 20
}


def calculate_bill(detected_objects):
    total_bill = 0
    for obj in detected_objects:
        obj_lower = obj.lower()
        if obj_lower in item_prices:
            total_bill += item_prices[obj_lower]
    return total_bill


def detect_objects(image):
    model = YOLO('best 80.pt')
    results = model.predict(source=image)
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
    st.markdown('<h3>Welcome!</h3>', unsafe_allow_html=True)

    capture_button = st.button("Capture Image from Webcam")
    if capture_button:
        image = capture_image()
        if image is not None:
            st.image(image, caption='Captured Image', use_column_width=True)

            # Detect objects in the captured image
            detected_objects = detect_objects(image)

            # Display detected objects and their prices on the web page
            st.subheader("Detected Objects:")
            for obj in detected_objects:
                # Concatenate object name and price if available
                if obj.lower() in item_prices:
                    item_with_price = f"{obj} (₹{item_prices[obj.lower()]})"
                else:
                    item_with_price = obj
                # Display object name and price (if available) in the same row
                st.write(item_with_price)

            # Calculate total bill
            total_bill = calculate_bill(detected_objects)
            st.subheader("Total Bill:")
            st.write("₹", total_bill)


if __name__ == "__main__":
    app()
