import streamlit as st
import streamlit.components.v1 as components
import av
import time
import base64
import threading
from pathlib import Path

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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
# 2. HELPER: Audio Alert via Browser HTML5
# ==========================================
@st.cache_data
def get_audio_base64():
    """Encode file beep.wav ke base64 agar bisa dimainkan di browser."""
    audio_path = Path("assets/beep.wav")
    if audio_path.exists():
        audio_bytes = audio_path.read_bytes()
        return base64.b64encode(audio_bytes).decode()
    return None

def get_play_audio_html(audio_b64):
    """Menghasilkan HTML statis untuk memainkan audio.
    Karena string statis (tanpa timestamp), Streamlit tidak akan me-recreate
    iframe ini setiap detik. Script ini hanya jalan 1x saat status berubah ke Microsleep."""
    return f"""
    <script>
        var p = window.parent;
        if (p._msAudio) {{
            p._msAudio.pause();
        }}
        p._msAudio = new Audio("data:audio/wav;base64,{audio_b64}");
        p._msAudio.loop = true;
        p._msAudio.play().catch(function(e) {{ console.log("Audio play blocked/error:", e); }});
    </script>
    """

def get_stop_audio_html():
    """Menghasilkan HTML statis untuk menghentikan audio."""
    return """
    <script>
        var p = window.parent;
        if (p._msAudio) {
            p._msAudio.pause();
            p._msAudio = null;
        }
    </script>
    """

# ==========================================
# 3. SIDEBAR & KONTROL UI
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
# 4. WEBRTC VIDEO PROCESSOR
# ==========================================
class MicrosleepVideoProcessor(VideoProcessorBase):
    """
    Callback WebRTC yang berjalan di thread terpisah.
    Menerima frame dari browser, memproses deteksi microsleep,
    lalu mengembalikan frame dengan overlay visual.
    """
    def __init__(self):
        self.conf_threshold = 0.5
        self.alert_duration = 2.0
        self.start_sleep_time = None
        self._lock = threading.Lock()
        self._status = "Sadar"
        self._prev_time = time.time()

    @property
    def status(self):
        with self._lock:
            return self._status

    @status.setter
    def status(self, value):
        with self._lock:
            self._status = value

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Hitung FPS dari selisih waktu antar-frame
        curr_time = time.time()
        fps = 1.0 / (curr_time - self._prev_time) if (curr_time - self._prev_time) > 0 else 0.0
        self._prev_time = curr_time

        # Proses frame: deteksi wajah + klasifikasi kantuk + overlay visual
        processed_frame, current_status, self.start_sleep_time = process_frame_local(
            img,
            self.conf_threshold,
            self.alert_duration,
            self.start_sleep_time,
            fps=fps
        )

        self.status = current_status

        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

# ==========================================
# 5. MAIN UI & LOGIC
# ==========================================
st.title("👁️ Sistem Deteksi Microsleep Real-Time")
st.markdown("**Proyek Akhir Machine Learning** | YOLOv26 (Deteksi Wajah) + MobileNetV3 (Klasifikasi Kantuk)")
st.markdown("---")

col1, col2 = st.columns([3, 1])

with col1:
    st.write("### Live Camera Feed")
    
    # Konfigurasi ICE Server untuk WebRTC (STUN gratis dari Google)
    RTC_CONFIGURATION = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }

    ctx = webrtc_streamer(
        key="microsleep-detection",
        video_processor_factory=MicrosleepVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # ==========================================
    # 6. STATUS MONITOR (auto-refresh setiap 1 detik)
    # ==========================================
    @st.fragment(run_every=1)
    def status_monitor():
        """
        Fragment yang berjalan otomatis setiap 1 detik (tanpa me-restart WebRTC).
        Mengecek status dari video processor dan mengontrol audio alarm.
        """
        if ctx.video_processor:
            # Sinkronkan setting dari sidebar ke processor
            ctx.video_processor.conf_threshold = st.session_state.global_conf_threshold
            ctx.video_processor.alert_duration = st.session_state.global_alert_duration

            # Cek status terkini dari processor thread
            current_status = ctx.video_processor.status
            if current_status == "Microsleep":
                st.error("⚠️ PERINGATAN: Pengemudi terdeteksi Microsleep!")
                audio_b64 = get_audio_base64()
                if audio_b64:
                    components.html(get_play_audio_html(audio_b64), height=0)
            else:
                st.success("✅ Pengemudi dalam keadaan Sadar.")
                components.html(get_stop_audio_html(), height=0)
        else:
            st.info("Klik **START** untuk memulai deteksi melalui webcam.")

    status_monitor()

with col2:
    st.write("### Panduan")
    st.info("""
    1. Login sebagai Admin untuk mengakses kontrol panel.
    2. Klik **START** untuk memulai deteksi webcam.
    3. Izinkan akses kamera pada browser.
    4. Sistem akan mendeteksi wajah dan mengklasifikasikan kantuk.
    5. Jika durasi tertidur > ambang batas, peringatan akan muncul.
    """)
    st.warning("⚠️ Pastikan browser mengizinkan akses kamera.")
    st.success("Deteksi visual (bounding box, label, warning) langsung tampil pada video feed secara real-time.")

