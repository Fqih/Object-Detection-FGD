import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

model = YOLO("Model/YoloV11.pt")

def predict_image(img):
    results = model(img)
    return results

def draw_boxes(img, results):
    img = np.array(img)
    detected_classes = set()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_y_position = y1 - 10
            if label_y_position < 10:
                label_y_position = y2 + 20

            cv2.putText(img, label, (x1, label_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected_classes.add(model.names[cls])
    
    return img, detected_classes

st.title("Deteksi Objek dengan YOLO 🚀")

option = st.selectbox("Pilih Metode Input 📸", ["Upload Gambar", "Gunakan Kamera"])

if option == "Upload Gambar":
    uploaded_file = st.file_uploader("Unggah Gambar... 📥", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Gambar Yang Diupload 🖼️', use_column_width=True)

        results = predict_image(image)
        img_with_boxes, detected_classes = draw_boxes(image, results)

        st.image(img_with_boxes, caption='Objek yang Terdeteksi 🟢', use_column_width=True)

        if detected_classes:
            st.subheader("Kelas yang Terdeteksi 🏷️:")
            detected_classes_list = "\n".join([f"- {cls}" for cls in detected_classes])
            st.markdown(detected_classes_list)
        else:
            st.write("Tidak ada objek yang terdeteksi. :)")

elif option == "Gunakan Kamera":
    camera_input = st.camera_input("Ambil Gambar 📸")

    if camera_input is not None:
        image = Image.open(camera_input).convert('RGB')
        results = predict_image(image)
        img_with_boxes, detected_classes = draw_boxes(image, results)

        st.image(img_with_boxes, caption='Objek yang Terdeteksi 🟢', use_column_width=True)

        if detected_classes:
            st.subheader("Kelas yang Terdeteksi 🏷️:")
            st.write(", ".join(detected_classes))
        else:
            st.write("Tidak ada objek yang terdeteksi. :(")
