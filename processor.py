import cv2
import torch
import time
try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
from models_handler import load_models, get_transform

class_names = ['Sadar', 'Microsleep']

# ── Cache referensi model & transform di level modul ────────────────────────
# Hindari overhead function call + lookup cache setiap frame
_yolo_model = None
_mobilenet_model = None
_device = None
_transform = None

def _init_models():
    """Load dan cache semua model sekali saja."""
    global _yolo_model, _mobilenet_model, _device, _transform
    if _yolo_model is None:
        _yolo_model, _mobilenet_model, _device = load_models()
        _transform = get_transform()

def get_yolo_and_mobilenet():
    _init_models()
    return _yolo_model, _mobilenet_model, _device

# ── Ukuran inference YOLO (lebih kecil = lebih cepat) ───────────────────────
YOLO_IMGSZ = 640   # default 640 → turunkan ke 320 untuk 2-4x lebih cepat di CPU
DISPLAY_W  = 640   # lebar frame yang ditampilkan ke browser


# ════════════════════════════════════════════════════════════════════════════
# VideoProcessor  –  Mode WebRTC
# ════════════════════════════════════════════════════════════════════════════
class VideoProcessor:
    def __init__(self):
        self.start_sleep_time = None
        self.conf_threshold   = 0.5
        self.alert_duration   = 2.0
        self.last_time        = time.time()
        self._frame_count     = 0
        self._last_label      = "Sadar"
        self._last_color      = (0, 255, 0)
        self._last_boxes      = []   # simpan hasil deteksi terakhir untuk frame-skip

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        _init_models()

        self._frame_count += 1
        run_inference = (self._frame_count % 2 == 0)   # inference setiap 2 frame

        eyes_closed = False

        if run_inference:
            results = _yolo_model(img, verbose=False,
                                  conf=self.conf_threshold,
                                  imgsz=YOLO_IMGSZ)[0]
            self._last_boxes = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    tensor = _transform(crop).unsqueeze(0).to(_device)
                    with torch.no_grad():
                        _, pred = torch.max(_mobilenet_model(tensor), 1)
                    label = class_names[pred.item()]
                    color = (0, 0, 255) if label == 'Microsleep' else (0, 255, 0)
                    self._last_boxes.append((x1, y1, x2, y2, label, color))
                    if label == 'Microsleep':
                        eyes_closed = True
        else:
            # Gunakan hasil frame sebelumnya
            for *_, label, _ in self._last_boxes:
                if label == 'Microsleep':
                    eyes_closed = True
                    break

        # Gambar box dari hasil terbaru
        for x1, y1, x2, y2, label, color in self._last_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        current_time = time.time()
        if eyes_closed:
            if self.start_sleep_time is None:
                self.start_sleep_time = current_time
            elapsed = current_time - self.start_sleep_time
            cv2.putText(img, f"Microsleep: {elapsed:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            if elapsed >= self.alert_duration:
                cv2.putText(img, "BAHAYA! BANGUN!", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)
        else:
            self.start_sleep_time = None
            cv2.putText(img, "Status: AMAN", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        fps = 1.0 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0.0
        self.last_time = current_time
        cv2.putText(img, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ════════════════════════════════════════════════════════════════════════════
# process_frame_local  –  Mode OpenCV lokal
# ════════════════════════════════════════════════════════════════════════════
def process_frame_local(frame, conf_threshold, alert_duration, start_sleep_time,
                        frame_count=0):
    """
    Proses satu frame webcam lokal.
    - Inference YOLO di resolusi YOLO_IMGSZ (lebih kecil = lebih cepat).
    - MobileNet hanya dijalankan bila ada deteksi wajah.
    """
    _init_models()

    # Resize frame kecil khusus untuk YOLO (tanpa mengubah frame asli untuk display)
    h_orig, w_orig = frame.shape[:2]
    scale = YOLO_IMGSZ / max(h_orig, w_orig)
    if scale < 1.0:
        small = cv2.resize(frame, (int(w_orig * scale), int(h_orig * scale)))
    else:
        small = frame

    results = _yolo_model(small, verbose=False, conf=conf_threshold, imgsz=YOLO_IMGSZ)[0]
    current_status = "Sadar"
    eyes_closed    = False

    for box in results.boxes:
        # Skala balik koordinat ke resolusi asli
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        if scale < 1.0:
            x1 = int(x1 / scale); y1 = int(y1 / scale)
            x2 = int(x2 / scale); y2 = int(y2 / scale)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            tensor = _transform(crop).unsqueeze(0).to(_device)
            with torch.no_grad():
                _, pred = torch.max(_mobilenet_model(tensor), 1)
            label = class_names[pred.item()]
            if label == 'Microsleep':
                eyes_closed = True
            color = (0, 0, 255) if label == 'Microsleep' else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    current_time = time.time()
    if eyes_closed:
        if start_sleep_time is None:
            start_sleep_time = current_time
        elapsed = current_time - start_sleep_time
        cv2.putText(frame, f"Microsleep: {elapsed:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        if elapsed >= alert_duration:
            current_status = "Microsleep"
            cv2.putText(frame, "BAHAYA! BANGUN!", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
    else:
        start_sleep_time = None
        cv2.putText(frame, "Status: AMAN", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame, current_status, start_sleep_time
