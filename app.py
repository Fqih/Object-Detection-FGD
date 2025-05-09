import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
import time
import pandas as pd
import os
import traceback
from ultralytics import YOLO

# Siapkan page config
st.set_page_config(
    page_title="Deteksi Rambu Lalu Lintas",
    page_icon="🚦",
    layout="wide"
)

# Load model YOLO
@st.cache_resource
def load_yolo_models():
    model_paths = {
        "YOLOv8": "Model/YoloV8.pt",
        "YOLOv9": "Model/YoloV9.pt",
        "YOLOv10": "Model/YoloV10.pt",
        "YOLOv11": "Model/YoloV11.pt",
    }
    
    models = {}
    for name, path in model_paths.items():
        try:
            models[name] = YOLO(path)
        except Exception as e:
            pass
    
    return models

# Daftar nama kelas rambu
class_names = [
    'Balai Pertolongan Pertama', 'Banyak Anak-Anak', 'Banyak Tikungan Pertama Kanan', 'Banyak Tikungan Pertama Kiri',
    'Berhenti', 'Dilarang Belok Kanan', 'Dilarang Belok Kiri', 'Dilarang Berhenti', 'Dilarang Masuk',
    'Dilarang Mendahului', 'Dilarang Parkir', 'Dilarang Putar Balik', 'Gereja', 'Hati-Hati', 'Ikuti Bundaran',
    'Jalur Sepeda', 'Kecepatan Maks. 30 km', 'Kecepatan Maks. 40 km', 'Lajur Kiri', 'Lampu Lalu Lintas',
    'Larangan Muatan - 10 ton', 'Masjid', 'Pemberhentian Bus', 'Penyebrangan Pejalan Kaki',
    'Peringatan Perlintasan Kereta Api', 'Perintah Jalur Penyebrangan', 'Persimpangan 3 Prioritas',
    'Persimpangan 3 Prioritas Kanan', 'Persimpangan 3 Prioritas Kiri', 'Persimpangan 3 Sisi Kiri',
    'Persimpangan 4', 'Pilih Salah Satu Jalur', 'Polisi Tidur', 'Pom Bensin', 'Putar Balik',
    'Rumah Sakit', 'Tempat Parkir', 'Tikungan Ganda Pertama Ke Kanan', 'Tikungan Ganda Pertama Ke Kiri',
    'Tikungan Ke Kanan'
]

# Load model SSDLite
@st.cache_resource
def load_ssdlite_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 41  # 40 kelas + background
    
    # Coba beberapa pendekatan berbeda untuk memuat model
    model_file = "Model/SSDLite.pth"
    
    # Metode 1: Coba load model langsung
    try:
        model = torch.load(model_file, map_location=device)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e1:
        pass
    
    # Metode 2: Coba buat model dengan width_mult yang lebih kecil
    try:
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            pretrained=False, 
            width_mult=0.5  # Gunakan width_mult yang lebih kecil
        )
        model.num_classes = num_classes
        
        # Coba load state_dict dengan strict=False
        state_dict = torch.load(model_file, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e2:
        pass
    
    # Metode 3: Buat model kosong sebagai fallback
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=False)
    model.num_classes = num_classes
    model.to(device)
    model.eval()
    return model, device

# Fungsi untuk prediksi dengan model YOLO
def predict_yolo(model, img):
    start = time.time()
    results = model(img)
    end = time.time()
    return results, end - start

# Fungsi untuk prediksi dengan model SSDLite
def predict_ssdlite(model, img, device):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((300, 300)),
        torchvision.transforms.ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    start = time.time()
    try:
        with torch.no_grad():
            results = model(img_tensor)
        end = time.time()
        
        # Handle berbagai format output model
        if isinstance(results, list):
            result = results[0]
        else:
            result = results
            
        # Pastikan hasil memiliki kunci yang dibutuhkan
        if 'boxes' not in result or 'labels' not in result or 'scores' not in result:
            # Buat dummy result
            dummy_result = {
                'boxes': torch.tensor([], device=device),
                'labels': torch.tensor([], device=device),
                'scores': torch.tensor([], device=device)
            }
            return dummy_result, end - start
            
        return result, end - start
    except Exception as e:
        end = time.time()
        # Buat dummy result
        dummy_result = {
            'boxes': torch.tensor([], device=device),
            'labels': torch.tensor([], device=device),
            'scores': torch.tensor([], device=device)
        }
        return dummy_result, end - start

# Fungsi untuk menggambar bounding box dari hasil YOLO
def draw_yolo_boxes(img, results, model):
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
            label_y_position = y1 - 10 if y1 - 10 >= 10 else y2 + 20
            cv2.putText(img, label, (x1, label_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected_classes.add(model.names[cls])

    return img, detected_classes

# Fungsi untuk menggambar bounding box dari hasil SSDLite - hanya menampilkan kelas dengan confidence tertinggi
def draw_ssdlite_boxes(img, results):
    img = np.array(img.copy())
    h, w = img.shape[:2]
    
    # Cek apakah hasil valid
    if results['boxes'].numel() == 0:
        return img, set()
    
    try:
        boxes = results['boxes'].cpu().numpy()
        labels = results['labels'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        
        # Jika ada deteksi, cari yang memiliki confidence tertinggi
        if len(scores) > 0:
            # Cari indeks dengan nilai confidence tertinggi
            top_idx = np.argmax(scores)
            
            # Ambil data untuk deteksi dengan confidence tertinggi
            box = boxes[top_idx]
            label = labels[top_idx]
            score = scores[top_idx]
            
            # Proses bounding box
            x1, y1, x2, y2 = box
            
            # Scale boxes jika gambar diubah ukurannya
            x1 = max(0, int(x1 * w / 300))
            y1 = max(0, int(y1 * h / 300))
            x2 = min(w, int(x2 * w / 300))
            y2 = min(h, int(y2 * h / 300))
            
            # Pemetaan indeks ke nama kelas
            idx = (int(label) - 1) % len(class_names)
            class_name = class_names[idx]
            
            # Tampilkan informasi label, indeks asli, dan confidence
            label_text = f"{class_name} ({int(label)}) {score:.2f}"
            
            # Gambar bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_y_position = y1 - 10 if y1 - 10 >= 10 else y2 + 20
            cv2.putText(img, label_text, (x1, label_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Kembalikan gambar dan set dengan satu kelas yang terdeteksi
            return img, {class_name}
        
        # Jika tidak ada deteksi, kembalikan gambar asli dan set kosong
        return img, set()
    except Exception as e:
        # Jika terjadi kesalahan, kembalikan gambar asli dan set kosong
        return img, set()

# Fungsi utama untuk UI
def main():
    # Navigasi sidebar
    page = st.sidebar.selectbox("🧭 Navigasi", ["Deteksi Objek", "Tentang Aplikasi"])

    if page == "Tentang Aplikasi":
        st.title("Tentang Aplikasi")
        st.markdown("""
    Aplikasi ini adalah bentuk keluaran dari kegiatan **Forum Group Discussion (FGD) Asisten MK Praktikum Unggulan (Praktikum DGX) Universitas Gunadarma Semester Genap ATA 2024/2025**.

    ---

    ### 👨‍💻 Dibuat oleh:

    **Divisi Research:**
    - Muhamad Alif Ramadhan  
    - Nabilah Ismah Wiananda  
    - Tegar Sakti Ramadhan

    **Divisi Paper:**
    - Muhammad Daffa Alghifari  
    - Zahwa Annisa Hendajani

    **Divisi Competition:**
    - Danu Tirta  
    - Muhammad Faqih Hakim  
    - Nazwa Akilla Zahra

    **Pendamping:**
    - Prof. Dr. Detty Purnamasari, S.Kom., MMSI., M.I.Kom.  
    - Ulfa Hidayati, S.T., MMSI.  
    - Milda Safrila Oktiana, S.Kom., MMSI.  
    - Fanka Arie Reza, S.Kom.  
    - Mario Mora Siregar

    ---

    ### Penjelasan Aplikasi:

    Aplikasi ini merupakan website deteksi objek berbasis **deep learning** untuk mengenali dan mengklasifikasikan berbagai jenis **rambu lalu lintas** dari gambar yang diunggah pengguna.  

    Dibangun menggunakan framework **Streamlit**, aplikasi ini menyajikan antarmuka yang interaktif dan mudah digunakan. Pengguna dapat:
    - Mengunggah gambar atau menggunakan kamera
    - Memilih model deteksi objek yang tersedia
    - Melihat hasil deteksi secara langsung

    Model deteksi yang digunakan menerapkan pendekatan **one-stage detection**, yang langsung memprediksi posisi dan kelas objek dari gambar input.

    ---

    ### 🔍 Model yang Digunakan:

    1. **YOLOv8n**  
    2. **YOLOv9t**  
    3. **YOLOv10n**  
    4. **YOLOv11n**  
    5. **SSDLite (Single Shot MultiBox Detector)**

    Setiap model memiliki kekuatan berbeda dalam hal **kecepatan dan akurasi**, dan pengguna bebas memilih model yang sesuai.

    > ⚠️ **Catatan:** Model memiliki keterbatasan dalam mendeteksi objek dengan jarak lebih dari 50 meter karena keterbatasan resolusi dan skala objek kecil.

    ---

    ### 🛑 Daftar Kelas yang Dideteksi (40 kelas):

    - Balai Pertolongan Pertama  
    - Banyak Anak-Anak  
    - Banyak Tikungan Pertama Kanan  
    - Banyak Tikungan Pertama Kiri  
    - Berhenti  
    - Dilarang Belok Kanan  
    - Dilarang Belok Kiri  
    - Dilarang Berhenti  
    - Dilarang Masuk  
    - Dilarang Mendahului  
    - Dilarang Parkir  
    - Dilarang Putar Balik  
    - Gereja  
    - Hati-Hati  
    - Ikuti Bundaran  
    - Jalur Sepeda  
    - Kecepatan Maks. 30 km  
    - Kecepatan Maks. 40 km  
    - Lajur Kiri  
    - Lampu Lalu Lintas  
    - Larangan Muatan > 10 Ton  
    - Masjid  
    - Pemberhentian Bus  
    - Penyebrangan Pejalan Kaki  
    - Peringatan Perlintasan Kereta Api  
    - Perintah Jalur Penyebrangan  
    - Persimpangan 3 Prioritas  
    - Perimpangan 3 Prioritas Kanan  
    - Persimpangan 3 Prioritas Kiri  
    - Persimpangan 3 Sisi Kiri  
    - Perimpangan 4  
    - Pilih Salah Satu Jalur  
    - Polisi Tidur  
    - Pom Bensin  
    - Putar Balik  
    - Rumah Sakit  
    - Tempat Parkir  
    - Tikungan Ganda Pertama ke Kanan  
    - Tikungan Ganda Pertama ke Kiri  
    - Tikungan ke Kanan

    ---
    """)


    else:
        st.title("Deteksi Objek Rambu Lalu Lintas 🚦")
        
        # Pemuatan model
        yolo_models = load_yolo_models()
        
        # Tambahkan SSDLite ke pilihan model hanya jika ada model YOLO yang berhasil dimuat
        if yolo_models:
            try:
                ssdlite_model, device = load_ssdlite_model()
                has_ssdlite = True
            except Exception as e:
                has_ssdlite = False
        else:
            return
        
        # Pilihan model
        model_options = list(yolo_models.keys())
        if has_ssdlite:
            model_options.append("SSDLite")
        model_options.append("Bandingkan Semua Model")
        
        model_choice = st.selectbox("Pilih Model 🎭", model_options)

        # Input gambar
        input_method = st.selectbox("Pilih Metode Input 📸", ["Upload Gambar", "Gunakan Kamera"])
        image = None

        if input_method == "Upload Gambar":
            uploaded_file = st.file_uploader("Unggah Gambar... 📥", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption='Gambar Yang Diupload 🖼️', use_column_width=True)

        elif input_method == "Gunakan Kamera":
            camera_input = st.camera_input("Ambil Gambar 📸")
            if camera_input:
                image = Image.open(camera_input).convert("RGB")

        # Proses deteksi jika gambar tersedia
        if image:
            if model_choice == "Bandingkan Semua Model":
                result_images = {}
                times = {}
                detected_sets = {}

                # Deteksi dengan model YOLO
                for name, model in yolo_models.items():
                    try:
                        result, duration = predict_yolo(model, image)
                        img_boxed, detected = draw_yolo_boxes(image.copy(), result, model)
                        result_images[name] = img_boxed
                        times[name] = duration
                        detected_sets[name] = detected
                    except Exception as e:
                        result_images[name] = np.array(image.copy())
                        times[name] = 0.0
                        detected_sets[name] = set()
                
                # Deteksi dengan model SSDLite jika tersedia
                if has_ssdlite:
                    try:
                        result, duration = predict_ssdlite(ssdlite_model, image, device)
                        img_boxed, detected = draw_ssdlite_boxes(image, result)
                        result_images["SSDLite"] = img_boxed
                        times["SSDLite"] = duration
                        detected_sets["SSDLite"] = detected
                    except Exception as e:
                        result_images["SSDLite"] = np.array(image.copy())
                        times["SSDLite"] = 0.0
                        detected_sets["SSDLite"] = set()
                
                # Tampilkan hasil dari semua model
                st.subheader("Hasil Deteksi 🔍")
                
                # Buat layout columns
                cols = st.columns(len(result_images))
                
                # Tampilkan setiap hasil
                for i, (name, img) in enumerate(result_images.items()):
                    with cols[i]:
                        st.subheader(name)
                        st.image(img, caption=f"Hasil {name}", use_column_width=True)
                        st.write(f"⏱️ Waktu Inferensi: {times[name]:.4f} detik")
                        st.write("🏷️ Objek Terdeteksi:")
                        if detected_sets[name]:
                            for cls in detected_sets[name]:
                                st.write(f"- {cls}")
                        else:
                            st.write("- Tidak ada objek terdeteksi")
                
                # Tampilkan tabel perbandingan
                st.subheader("Perbandingan Performa Model 📊")
                
                model_names = list(times.keys())
                inference_times = list(times.values())
                num_objects = [len(detected) for detected in detected_sets.values()]
                
                comparison_data = {
                    "Model": model_names,
                    "Waktu Inferensi (detik)": inference_times,
                    "Jumlah Objek Terdeteksi": num_objects
                }
                
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
                
                # Visualisasi perbandingan waktu inferensi
                st.subheader("Grafik Waktu Inferensi 📈")
                chart_data = pd.DataFrame({
                    "Model": model_names,
                    "Waktu (detik)": inference_times
                })
                st.bar_chart(chart_data.set_index("Model"))
                
            else:  # Jika hanya satu model yang dipilih
                if model_choice in yolo_models:
                    try:
                        result, duration = predict_yolo(yolo_models[model_choice], image)
                        img_boxed, detected = draw_yolo_boxes(image.copy(), result, yolo_models[model_choice])
                        
                        st.subheader("Hasil Deteksi 🔍")
                        st.image(img_boxed, caption=f"Hasil Deteksi {model_choice}", use_column_width=True)
                        
                        # Tampilkan waktu inferensi dan objek terdeteksi
                        st.write(f"⏱️ Waktu Inferensi: {duration:.4f} detik")
                        st.write("🏷️ Objek Terdeteksi:")
                        if detected:
                            for cls in detected:
                                st.write(f"- {cls}")
                        else:
                            st.write("- Tidak ada objek terdeteksi")
                    except Exception as e:
                        pass
                
                elif model_choice == "SSDLite" and has_ssdlite:
                    try:
                        result, duration = predict_ssdlite(ssdlite_model, image, device)
                        img_boxed, detected = draw_ssdlite_boxes(image, result)
                        
                        st.subheader("Hasil Deteksi 🔍")
                        st.image(img_boxed, caption="Hasil Deteksi SSDLite", use_column_width=True)
                        
                        # Tampilkan waktu inferensi dan objek terdeteksi
                        st.write(f"⏱️ Waktu Inferensi: {duration:.4f} detik")
                        st.write("🏷️ Objek Terdeteksi:")
                        if detected:
                            for cls in detected:
                                st.write(f"- {cls}")
                        else:
                            st.write("- Tidak ada objek terdeteksi")
                    except Exception as e:
                        pass


        # Tambahkan bantuan dan petunjuk
        with st.sidebar:
            st.subheader("Petunjuk Penggunaan 📝")
            st.markdown("""
            1. Pilih model deteksi yang ingin digunakan
            2. Unggah gambar atau ambil foto dengan kamera
            3. Hasil deteksi akan ditampilkan beserta informasi waktu inferensi
            4. Untuk membandingkan model, pilih "Bandingkan Semua Model"
            """)
            
            # Informasi sistem
            st.sidebar.write(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            if torch.cuda.is_available():
                st.sidebar.write(f"GPU: {torch.cuda.get_device_name(0)}")
        
        footer = """
        <style>
        .footer {
            background-color: var(--secondary-background-color);
            width: 100%;
            text-align: center;
            font-size: 14px;
            padding: 15px 0 10px 0;
            margin-top: 50px;
            border-top: 1px solid #eaeaea;
        }
        </style>
        <div class="footer">
            Universitas Gunadarma<br>
            Mei 2025
        </div>
        """

        st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
