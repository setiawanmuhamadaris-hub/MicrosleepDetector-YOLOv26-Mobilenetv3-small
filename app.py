import streamlit as st
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import ultralytics.nn.modules.block as block
from ultralytics import YOLO
from torchvision.models import mobilenet_v3_small
import numpy as np
import time
import threading
import winsound

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(page_title="Deteksi Microsleep", page_icon="👁️", layout="wide")
st.title("👁️ Sistem Deteksi Microsleep Real-Time")
st.markdown("**Proyek Akhir Machine Learning** | YOLOv26 (Deteksi Wajah) + MobileNetV3 (Klasifikasi Kantuk)")

# ==========================================
# 2. MONKEY PATCHING ARSITEKTUR (WAJIB)
# ==========================================
class AttentionEnhancedBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = block.Conv(c1, c_, k[0], 1)
        self.cv2 = block.Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2 // 4, c2, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = out * self.se(out)
        return x + out if self.add else out

# Terapkan patch secara global
block.Bottleneck = AttentionEnhancedBottleneck

# ==========================================
# 3. CACHING MODEL (AGAR STREAMLIT TIDAK LEMOT)
# ==========================================
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load YOLOv26
    yolo = YOLO('models/best.pt') 
    
    # Load MobileNetV3
    mobilenet = mobilenet_v3_small()
    mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, 2)
    # Ganti path ini ke lokasi file mobilenetv3_best.pth hasil downloadmu
    mobilenet.load_state_dict(torch.load('models/mobilenetv3_best.pth', map_location=device, weights_only=True))
    mobilenet = mobilenet.to(device)
    mobilenet.eval()
    
    return yolo, mobilenet, device

# Muat model ke memori
yolo_model, mobilenet_model, device = load_models()

# Transformasi MobileNet
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class_names = ['Sadar', 'Microsleep']

# ==========================================
# 4. SIDEBAR & KONTROL UI
# ==========================================
st.sidebar.header("Kontrol Deteksi")
conf_threshold = 0.5 # Ditetapkan 0.5 sesuai permintaan
run_webcam = st.sidebar.checkbox("Mulai Kamera 🎥")

st.sidebar.markdown("---")
st.sidebar.info("Pastikan wajah terlihat jelas oleh kamera agar bounding box dapat memotong area dengan akurat.")

# ==========================================
# 5. MAIN LOOP WEBCAM
# ==========================================
# Placeholder untuk menampilkan frame video
frame_window = st.image([])
status_text = st.empty()

if run_webcam:
    cap = cv2.VideoCapture(0) # Gunakan ID 0 untuk webcam bawaan
    last_beep_time = 0
    
    while run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal membaca sinyal kamera.")
            break
            
        # YOLO Detection
        results = yolo_model(frame, verbose=False, conf=conf_threshold)[0]
        
        current_status = "Tidak ada wajah terdeteksi"
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop_img = frame[y1:y2, x1:x2]
            
            if crop_img.size > 0:
                # Klasifikasi MobileNet
                input_tensor = transform(crop_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = mobilenet_model(input_tensor)
                    _, preds = torch.max(outputs, 1)
                    label = class_names[preds.item()]
                    
                current_status = label
                color = (0, 0, 255) if label == 'Microsleep' else (0, 255, 0)
                
                # Visualisasi Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Konversi warna OpenCV (BGR) ke format Streamlit (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)
        
        # Tampilkan status di bawah kamera
        if current_status == "Microsleep":
            status_text.error("⚠️ PERINGATAN: Pengemudi terdeteksi Microsleep!")
            
            # Bunyikan alarm beep tiap 0.5 detik tanpa memblokir frame video
            current_time = time.time()
            if current_time - last_beep_time > 0.5:
                winsound.PlaySound('assets/beep.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
                last_beep_time = current_time
                
        elif current_status == "Sadar":
            status_text.success("✅ Pengemudi dalam keadaan Sadar.")
        else:
            status_text.warning("Menunggu deteksi wajah...")
            
    cap.release()
else:
    st.info("Klik centang 'Mulai Kamera' di sidebar untuk memulai deteksi.")