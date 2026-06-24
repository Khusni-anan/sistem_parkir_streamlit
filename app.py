import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Dashboard Deteksi Lahan Parkir YOLOv11",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Menambahkan custom CSS untuk mempercantik tampilan dashboard
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #1E3A8A;
    }
    h1 {
        color: #1E3A8A;
        font-weight: 700;
    }
    h2 {
        color: #2563EB;
    }
    </style>
""", unsafe_allow_html=True)

# Judul Utama Dashboard
st.title("🚗 Dashboard Deteksi Lahan Parkir Kosong (YOLOv11)")
st.markdown("""
Selamat datang di Aplikasi Deteksi Lahan Parkir Otomatis!  
Aplikasi ini dikembangkan menggunakan **Streamlit** dan didukung oleh model **Ultralytics YOLOv11** (`best.pt`). 
Anda dapat mengunggah **Gambar** maupun **Video** CCTV lahan parkir untuk menganalisis ketersediaan slot parkir secara real-time.
""")

# ==========================================
# SIDEBAR - PENGATURAN & KONTROL MODEL
# ==========================================
st.sidebar.header("⚙️ Pengaturan Model")

# Input lokasi file weights (.pt)
model_path = st.sidebar.text_input("Path Model Weights (.pt)", value="best.pt")

# Loader model menggunakan cache agar memori efisien
@st.cache_resource
def load_yolo_model(path):
    if os.path.exists(path):
        try:
            return YOLO(path)
        except Exception as e:
            st.sidebar.error(f"Gagal memuat model: {e}")
            return None
    return None

model = load_yolo_model(model_path)

if model is not None:
    st.sidebar.success("✅ Model YOLOv11 Berhasil Dimuat!")
    # Tampilkan informasi kelas yang terdeteksi oleh model
    class_names = model.names
    st.sidebar.markdown("### 📋 Daftar Kelas Model:")
    for idx, name in class_names.items():
        st.sidebar.markdown(f"- Kelas {idx}: **`{name}`**")
else:
    st.sidebar.error(f"❌ Model tidak ditemukan di: '{model_path}'")
    st.sidebar.warning("Pastikan file 'best.pt' berada di folder yang sama dengan skrip ini atau masukkan path yang benar.")

# Slider untuk Threshold Kepercayaan (Confidence Score)
conf_threshold = st.sidebar.slider(
    "Confidence Threshold (Batasan Keyakinan)", 
    min_value=0.05, 
    max_value=1.00, 
    value=0.25, 
    step=0.05,
    help="Semakin kecil nilainya, semakin sensitif deteksinya. Semakin besar nilainya, semakin selektif deteksinya."
)

st.sidebar.markdown("---")
# Menu Navigasi Media
menu = st.sidebar.radio("📁 Pilih Media Input:", ["📸 Unggah Gambar", "🎥 Unggah Video"])

# ==========================================
# PROSES UTAMA DETEKSI
# ==========================================
if model is not None:
    
    if menu == "📸 Unggah Gambar":
        st.header("📸 Fitur Deteksi dari Gambar")
        st.write("Silakan unggah foto area lahan parkir Anda di bawah ini.")
        
        uploaded_image = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png", "webp"])
        
        if uploaded_image is not None:
            # Membuka Gambar
            image = Image.open(uploaded_image)
            
            with st.spinner("Sedang menganalisis gambar dengan YOLOv11..."):
                # Jalankan Inferensi YOLO
                results = model.predict(source=image, conf=conf_threshold)
                
                # Gambar bounding box hasil prediksi
                res_plotted = results[0].plot()
                # Konversi dari BGR (OpenCV) ke RGB (Streamlit)
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Menghitung jumlah objek per kelas secara dinamis
                boxes = results[0].boxes
                counts = {}
                for c in boxes.cls:
                    c_idx = int(c)
                    c_name = class_names.get(c_idx, f"Kelas_{c_idx}").lower()
                    counts[c_name] = counts.get(c_name, 0) + 1
            
            # Sinkronisasi nama kelas yang fleksibel (mobil vs lahan kosong)
            car_count = 0
            free_count = 0
            for name, count in counts.items():
                if 'car' in name or 'mobil' in name:
                    car_count += count
                elif 'free' in name or 'lahan' in name or 'kosong' in name:
                    free_count += count
            
            # Menampilkan Panel Metrik Statistik Ringkas
            st.markdown("### 📊 Statistik Slot Parkir")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Total Deteksi Objek", value=len(boxes))
            with col2:
                st.metric(label="🚗 Slot Terisi (Mobil)", value=car_count)
            with col3:
                st.metric(label="🟢 Slot Parkir Kosong (Free)", value=free_count, delta=f"{free_count} Tersedia", delta_color="normal")
            
            st.markdown("---")
            
            # Menampilkan Komparasi Gambar Sebelum & Sesudah
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(image, caption="Gambar Asli yang Diunggah", use_container_width=True)
            with col_img2:
                st.image(res_plotted_rgb, caption="Hasil Deteksi dan Analisis Lahan Parkir", use_container_width=True)

    elif menu == "🎥 Unggah Video":
        st.header("🎥 Fitur Deteksi dari Video")
        st.write("Silakan unggah rekaman video / CCTV area lahan parkir Anda.")
        
        uploaded_video = st.file_uploader("Pilih file video...", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_video is not None:
            # Membuat file temporary di sistem lokal agar OpenCV bisa membaca file path video tersebut
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            tfile.close()
            
            # Membuka video capture stream
            cap = cv2.VideoCapture(tfile.name)
            
            st.info("💡 Memproses video... Jalur visualisasi frame dan metrik akan diupdate secara langsung di bawah ini.")
            
            # Placeholder dinamis di Streamlit agar UI diperbarui tiap frame
            metric_placeholder = st.empty()
            frame_placeholder = st.empty()
            
            # Loop pembacaan frame video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break # Keluar loop jika video habis
                
                # Jalankan prediksi YOLO per frame (verbose=False agar log terminal bersih)
                results = model.predict(source=frame, conf=conf_threshold, verbose=False)
                
                # Plot anotasi bounding box pada frame
                res_plotted = results[0].plot()
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Hitung jumlah objek per frame
                boxes = results[0].boxes
                v_counts = {}
                for c in boxes.cls:
                    c_idx = int(c)
                    c_name = class_names.get(c_idx, f"Kelas_{c_idx}").lower()
                    v_counts[c_name] = v_counts.get(c_name, 0) + 1
                
                # Ekstrak data mobil dan slot kosong
                v_car_count = 0
                v_free_count = 0
                for name, count in v_counts.items():
                    if 'car' in name or 'mobil' in name:
                        v_car_count += count
                    elif 'free' in name or 'lahan' in name or 'kosong' in name:
                        v_free_count += count
                
                # Perbarui komponen metrik real-time
                with metric_placeholder.container():
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Total Deteksi Aktif", len(boxes))
                    m_col2.metric("🚗 Slot Terisi (Mobil)", v_car_count)
                    m_col3.metric("🟢 Slot Kosong Tersedia", v_free_count)
                
                # Perbarui tampilan gambar video frame real-time
                frame_placeholder.image(res_plotted_rgb, channels="RGB", use_container_width=True)
                
            # Bersihkan resource OpenCV dan hapus file temp setelah selesai
            cap.release()
            os.unlink(tfile.name)
            st.success("🎉 Pemrosesan video telah selesai sepenuhnya!")
            
else:
    st.info("Silakan siapkan file weights model `best.pt` Anda agar aplikasi dapat dijalankan.")
