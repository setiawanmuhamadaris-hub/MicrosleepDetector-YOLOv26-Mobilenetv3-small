import streamlit as st
import cv2
import time
import winsound

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

from auth import init_session_state, show_login_form
from models_handler import load_models, AttentionEnhancedBottleneck
from processor import VideoProcessor, process_frame_local, DISPLAY_W
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
    
    st.session_state.camera_mode = st.sidebar.radio(
        "Mode Kamera",
        ("Lokal (OpenCV)", "Cloud (WebRTC)")
    )
    
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
    
    if st.session_state.camera_mode == "Cloud (WebRTC)":
        if not WEBRTC_AVAILABLE:
            st.error("Modul 'streamlit-webrtc' tidak terinstal atau tidak dapat dimuat. Pastikan modul tersebut terinstal.")
        else:
            ctx = webrtc_streamer(
                key="driver-monitoring",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:stun2.l.google.com:19302"]},
                    ]
                }),
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if ctx.video_processor:
                ctx.video_processor.conf_threshold = st.session_state.global_conf_threshold
                ctx.video_processor.alert_duration = st.session_state.global_alert_duration
            
    else:
        # Mode Lokal (OpenCV)
        run_webcam = st.checkbox("Mulai Kamera Lokal 🎥")
        frame_window = st.empty()
        status_text = st.empty()
        
        if run_webcam:
            cap = cv2.VideoCapture(0)
            # Set resolusi kamera lebih rendah agar capture lebih cepat
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            last_beep_time  = 0
            start_sleep_time = None
            last_time        = time.time()
            prev_status      = None   # track perubahan status agar tidak update tiap frame
            frame_count      = 0

            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Gagal membaca sinyal kamera.")
                    break

                frame_count += 1
                processed_frame, current_status, start_sleep_time = process_frame_local(
                    frame,
                    st.session_state.global_conf_threshold,
                    st.session_state.global_alert_duration,
                    start_sleep_time,
                    frame_count
                )

                current_time = time.time()
                fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0.0
                last_time = current_time
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Resize frame sebelum dikirim ke browser (hemat bandwidth & rendering)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                if w > DISPLAY_W:
                    frame_rgb = cv2.resize(frame_rgb, (DISPLAY_W, int(h * DISPLAY_W / w)))
                frame_window.image(frame_rgb)

                # Update status hanya saat berubah (hemat Streamlit re-render)
                if current_status != prev_status:
                    if current_status == "Microsleep":
                        status_text.error("⚠️ PERINGATAN: Pengemudi terdeteksi Microsleep!")
                    else:
                        status_text.success("✅ Pengemudi dalam keadaan Sadar.")
                    prev_status = current_status

                if current_status == "Microsleep":
                    if current_time - last_beep_time > 0.5:
                        winsound.PlaySound('assets/beep.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
                        last_beep_time = current_time

            cap.release()
        else:
            st.info("Centang 'Mulai Kamera Lokal' untuk memulai deteksi.")

with col2:
    st.write("### Panduan")
    st.info("""
    1. Login sebagai Admin untuk mengakses kontrol panel.
    2. Pilih Mode Kamera: **Lokal (OpenCV)** atau **Cloud (WebRTC)**.
    3. Sistem akan mendeteksi wajah dan mengklasifikasikan kantuk.
    4. Jika durasi tertidur > ambang batas, alarm akan menyala.
    """)
    if st.session_state.camera_mode == "Cloud (WebRTC)":
        st.warning("Catatan: Mode Cloud (WebRTC) hanya mendukung peringatan visual karena keterbatasan browser untuk memutar audio secara otomatis.")
    else:
        st.success("Mode Lokal mendukung peringatan visual dan suara (beep).")
