import streamlit as st
import cv2
import time
import winsound

from auth import init_session_state, show_login_form
from models_handler import load_models, AttentionEnhancedBottleneck
from processor import process_frame_local
import sys

# Hack agar PyTorch bisa memuat weight yang sebelumnya dilatih di __main__ (app.py)
setattr(sys.modules['__main__'], 'AttentionEnhancedBottleneck', AttentionEnhancedBottleneck)

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(page_title="Deteksi Microsleep Real-Time", page_icon="👁️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div[data-testid="stMetricValue"] { color: #00ff00; }
    </style>
    """, unsafe_allow_html=True)

# Inisialisasi Session State
init_session_state()

# Pre-load models (Cached)
try:
    load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ==========================================
# 2. SIDEBAR & KONTROL UI
# ==========================================
if st.session_state.logged_in:
    st.sidebar.header("Control Panel")
    st.sidebar.success("✅ Logged in as Admin")
    
    st.session_state.global_conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.1, 1.0,
        st.session_state.global_conf_threshold,
    )
    st.session_state.global_alert_duration = st.sidebar.slider(
        "Durasi Alarm (detik)", 
        1.0, 5.0, 
        st.session_state.global_alert_duration,
    )
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()
else:
    if st.sidebar.button("🔑 Login Admin", use_container_width=True, type="primary"):
        st.session_state.show_login = True
        st.rerun()

if st.session_state.show_login and not st.session_state.logged_in:
    show_login_form()
    st.stop()

# ==========================================
# 3. MAIN UI & LOGIC
# ==========================================
st.title("👁️ Sistem Deteksi Microsleep Real-Time")
st.markdown("**Proyek Akhir Machine Learning** | YOLOv26 (Deteksi Wajah) + MobileNetV3 (Klasifikasi Kantuk)")
st.markdown("---")

col1, col2 = st.columns([3, 1])

with col1:
    st.write("### Live Camera Feed")
    
    run_webcam = st.checkbox("Mulai Kamera 🎥")
    frame_window = st.empty()
    status_text = st.empty()
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        last_beep_time = 0
        start_sleep_time = None
        prev_time = time.time()
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal membaca sinyal kamera.")
                break
            
            # Hitung FPS dari selisih waktu antar-frame
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time
            
            processed_frame, current_status, start_sleep_time = process_frame_local(
                frame, 
                st.session_state.global_conf_threshold, 
                st.session_state.global_alert_duration, 
                start_sleep_time,
                fps=fps
            )
            
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb)
            
            current_time = time.time()
            if current_status == "Microsleep":
                status_text.error("⚠️ PERINGATAN: Pengemudi terdeteksi Microsleep!")
                if current_time - last_beep_time > 0.5:
                    winsound.PlaySound('assets/beep.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
                    last_beep_time = current_time
            else:
                status_text.success("✅ Pengemudi dalam keadaan Sadar.")
                
        cap.release()
    else:
        st.info("Centang 'Mulai Kamera' untuk memulai deteksi.")

with col2:
    st.write("### Panduan")
    st.info("""
    1. Login sebagai Admin untuk mengakses kontrol panel.
    2. Klik **Mulai Kamera** untuk memulai deteksi.
    3. Sistem akan mendeteksi wajah dan mengklasifikasikan kantuk.
    4. Jika durasi tertidur > ambang batas, alarm akan menyala.
    """)
    st.success("Mode Lokal mendukung peringatan visual dan suara (beep).")
