<div align="center">

# 👁️ MicrosleepDetector

### Real-Time Drowsiness Detection System

**YOLOv26 (Face Detection) × MobileNetV3-small (Drowsiness Classification)**

[![Python](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv26](https://img.shields.io/badge/YOLOv26-Object%20Detection-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![WebRTC](https://img.shields.io/badge/WebRTC-Real--Time%20Camera-333333?style=for-the-badge&logo=webrtc&logoColor=white)](https://webrtc.org/)

<br/>

Sistem AI yang mendeteksi **microsleep** (kantuk mikro) pengemudi secara real-time melalui webcam.  
Menggabungkan dua model deep learning dalam arsitektur **two-stage pipeline** untuk deteksi yang cepat dan akurat.

**Proyek Akhir Machine Learning**

---

</div>

## 🎯 Apa itu Microsleep?

**Microsleep** adalah episode tidur singkat yang terjadi tanpa disadari, biasanya berlangsung **1–30 detik**. Pada pengemudi, microsleep sangat berbahaya karena dapat menyebabkan kecelakaan fatal. Menurut data, **20–30% kecelakaan lalu lintas** disebabkan oleh kantuk.

Sistem ini dirancang sebagai **early warning system** yang mendeteksi tanda-tanda kantuk melalui analisis wajah secara real-time dan memberikan peringatan sebelum microsleep terjadi.

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|---|---|
| 🧠 **Two-Stage AI Pipeline** | YOLOv26 mendeteksi wajah → MobileNetV3-small mengklasifikasikan kantuk |
| ⚡ **Real-Time Detection** | Inferensi langsung dari webcam via WebRTC dengan FPS counter |
| ⏱️ **Temporal Logic** | Bukan sekadar deteksi per-frame — sistem menghitung durasi kantuk untuk mengurangi false positive |
| 🚨 **Multi-Alert System** | Peringatan visual (bounding box merah, teks "BAHAYA!", border merah) + audio alarm di browser |
| 🎛️ **Admin Control Panel** | Atur Confidence Threshold & Durasi Alarm secara dinamis tanpa restart |
| 📊 **Live Overlay** | FPS, status deteksi, countdown timer, dan label klasifikasi langsung di frame kamera |
| 🔐 **Access Control** | Sistem login admin untuk mengakses panel konfigurasi |
| 🌐 **Cloud Ready** | Bisa di-deploy ke Streamlit Community Cloud — akses dari mana saja via browser |

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        MICROSLEEP DETECTION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  📷 Browser ──► ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    │
│   (WebRTC)      │   YOLOv26    │    │  Crop Face   │    │  MobileNetV3     │    │
│                 │  Detection   │───►│    ROI       │───►│  Classification  │    │
│                 │  (best.pt)   │    │  (224×224)   │    │  Sadar/Microsleep│    │
│                 └──────────────┘    └──────────────┘    └────────┬─────────┘    │
│                                                                  │              │
│                                                    ┌─────────────▼──────────┐  │
│                                                    │   Temporal Logic       │  │
│                                                    │   (Duration Tracker)   │  │
│                                                    └─────────────┬──────────┘  │
│                                                                  │              │
│                              ┌────────────────┐    ┌─────────────▼──────────┐  │
│                              │  🔊 Browser     │◄───│   Alert Decision      │  │
│                              │  Audio Alert    │    │   (threshold check)   │  │
│                              └────────────────┘    └────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Detail Alur per Frame:

1. **Capture** — Frame diambil dari webcam browser pengguna melalui WebRTC
2. **Face Detection** — YOLOv26 mendeteksi wajah dengan bounding box + confidence score
3. **Crop & Preprocess** — ROI wajah di-crop, di-resize ke 224×224, dan dinormalisasi menggunakan statistik ImageNet
4. **Classification** — MobileNetV3-small melakukan binary classification: `Sadar` vs `Microsleep`
5. **Temporal Analysis** — Jika kantuk terdeteksi selama lebih dari durasi threshold, alarm diaktifkan
6. **Rendering** — Overlay visual (bounding box, label, FPS, warning) di-render ke frame

---

## 🧠 Tentang Model AI

### YOLOv26 — Face Detection

Model deteksi objek berbasis YOLO yang dimodifikasi dengan **Attention-Enhanced Bottleneck**. Modifikasi ini menambahkan **Squeeze-and-Excitation (SE) module** pada setiap bottleneck block, memungkinkan model untuk memberikan bobot lebih tinggi pada fitur-fitur penting (channel attention).

```
SE Block Architecture:
Input → AdaptiveAvgPool2d(1) → Conv1×1 (squeeze: c → c/4) → ReLU 
      → Conv1×1 (excite: c/4 → c) → Sigmoid → Channel-wise Multiplication
```

> **Mengapa SE Block?** Dengan attention mechanism, model bisa fokus pada fitur wajah yang paling relevan (mata, mulut) sambil menekan noise dari background, meningkatkan akurasi deteksi.

### MobileNetV3-small — Drowsiness Classification

Arsitektur CNN ringan yang dioptimasi untuk perangkat mobile/edge. Classifier layer terakhir dimodifikasi untuk output **2 kelas**:

| Kelas | Label | Deskripsi |
|---|---|---|
| 0 | `Sadar` | Pengemudi dalam keadaan sadar dan waspada |
| 1 | `Microsleep` | Pengemudi menunjukkan tanda-tanda kantuk |

> **Mengapa MobileNetV3-small?** Karena ringan (~6 MB), cepat inference-nya, dan tetap akurat — cocok untuk aplikasi real-time yang membutuhkan low latency.

---

## 🛠️ Tech Stack

| Kategori | Teknologi | Fungsi |
|---|---|---|
| **Web Framework** | Streamlit | Antarmuka web interaktif dan real-time dashboard |
| **Face Detection** | YOLOv26 (Ultralytics) | Deteksi wajah dengan custom attention bottleneck |
| **Classification** | MobileNetV3-small (PyTorch) | Klasifikasi binary: Sadar vs Microsleep |
| **Camera Streaming** | streamlit-webrtc | Streaming webcam real-time via browser (WebRTC) |
| **Image Processing** | OpenCV (headless) | Drawing overlay: bounding box, label, FPS |
| **Deep Learning** | PyTorch + Torchvision | Inferensi model dan image transforms |
| **Audio Alert** | HTML5 Audio API | Alarm suara cross-platform di browser |
| **Language** | Python 3.9–3.11 | Backend dan logic utama |

---

## 📂 Struktur Proyek

```
MicrosleepDetector-YOLOv26-Mobilenetv3-small/
│
├── 📄 app.py                  # Entry point - Streamlit UI & WebRTC streamer
├── 📄 models_handler.py       # Model loading, caching, & custom SE Bottleneck
├── 📄 processor.py            # Frame processing: detection → classification → overlay
├── 📄 auth.py                 # Sistem autentikasi admin
├── 📄 requirements.txt        # Daftar dependensi Python
├── 📄 packages.txt            # System dependencies untuk Streamlit Cloud
├── 📄 .gitignore              # File & folder yang diabaikan Git
│
├── 📁 .streamlit/
│   └── config.toml            # Konfigurasi Streamlit server
│
├── 📁 models/
│   ├── best.pt                # Weight YOLOv26 untuk face detection (~6 MB)
│   └── mobilenetv3_best.pth   # Weight MobileNetV3-small untuk klasifikasi (~6 MB)
│
└── 📁 assets/
    └── beep.wav               # File audio alarm peringatan
```

---

## 🚀 Instalasi & Menjalankan Aplikasi

### Prasyarat

- **Python** 3.9, 3.10, atau 3.11
- **Webcam** yang terhubung ke komputer
- **Browser modern** (Chrome, Edge, Firefox) dengan akses kamera diizinkan
- **GPU** (opsional, mendukung CUDA untuk inferensi lebih cepat)

### Langkah 1 — Clone Repository

```bash
git clone https://github.com/setiawanmuhamadaris-hub/MicrosleepDetector-YOLOv26-Mobilenetv3-small.git
cd MicrosleepDetector-YOLOv26-Mobilenetv3-small
```

### Langkah 2 — Buat & Aktifkan Virtual Environment

```powershell
# Buat virtual environment
python -m venv env-mocrosleep

# Aktifkan (PowerShell)
.\env-mocrosleep\Scripts\activate

# Aktifkan (Command Prompt)
env-mocrosleep\Scripts\activate.bat

# Aktifkan (Linux/Mac)
source env-mocrosleep/bin/activate
```

### Langkah 3 — Install Dependensi

```powershell
pip install -r requirements.txt
```

<details>
<summary>📦 <strong>Daftar Dependensi</strong></summary>

| Package | Fungsi |
|---|---|
| `streamlit` | Web framework untuk UI |
| `streamlit-webrtc` | Streaming webcam real-time via WebRTC |
| `opencv-python-headless` | Image processing & drawing overlay |
| `torch` | PyTorch deep learning framework |
| `torchvision` | Pre-trained models & transforms |
| `ultralytics` | YOLO framework |
| `numpy` | Operasi numerik |
| `av` | PyAV - video frame handling untuk WebRTC |

</details>

### Langkah 4 — Jalankan Aplikasi

```powershell
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501` 🎉

---

## ☁️ Deploy ke Streamlit Community Cloud

### Langkah 1 — Push ke GitHub

Pastikan semua file sudah di-commit dan di-push ke repository GitHub:

```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### Langkah 2 — Deploy di Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan akun GitHub
3. Klik **"New app"**
4. Pilih repository, branch (`main`), dan file utama (`app.py`)
5. Klik **"Deploy!"**

### Catatan Penting untuk Cloud Deployment

| Item | Detail |
|---|---|
| **Model files** | File `.pt` dan `.pth` (~12 MB total) akan di-clone langsung dari repo |
| **System packages** | `packages.txt` otomatis diinstall oleh Streamlit Cloud |
| **Kamera** | Browser harus mengizinkan akses kamera (muncul popup permission) |
| **HTTPS** | Streamlit Cloud otomatis serve via HTTPS (dibutuhkan WebRTC) |

---

## 📖 Cara Penggunaan

### Mode Guest (Tanpa Login)

1. Jalankan aplikasi atau buka URL deployment
2. Klik **START** untuk memulai streaming webcam
3. Izinkan akses kamera pada browser
4. Posisikan wajah di depan webcam
5. Sistem akan langsung mendeteksi dan menampilkan status

### Mode Admin (Dengan Login)

1. Klik **"🔑 Login Admin"** di sidebar
2. Masukkan kredensial admin
3. Setelah login, panel kontrol akan muncul di sidebar:

| Kontrol | Range | Default | Fungsi |
|---|---|---|---|
| **Confidence Threshold** | 0.1 – 1.0 | 0.5 | Sensitivitas deteksi wajah (semakin rendah = semakin sensitif) |
| **Durasi Alarm** | 1.0 – 5.0 detik | 2.0 | Berapa lama kantuk harus terdeteksi sebelum alarm menyala |

### Indikator Visual

| Indikator | Arti |
|---|---|
| 🟢 Bounding box **hijau** + label "Sadar" | Pengemudi sadar dan waspada |
| 🔴 Bounding box **merah** + label "Microsleep" | Kantuk terdeteksi |
| ⏱️ **Countdown timer** (kuning) | Durasi kantuk yang sedang berjalan |
| 🚨 **"BAHAYA! BANGUN!"** + border merah | Microsleep melampaui threshold — alarm aktif! |
| 📊 **FPS counter** (kanan atas) | Kecepatan inferensi real-time |

---

## ⚙️ Konfigurasi

### Parameter yang Dapat Diubah (via Admin Panel)

| Parameter | Penjelasan | Tips |
|---|---|---|
| `Confidence Threshold` | Minimum confidence score agar YOLO menganggap deteksi valid | Turunkan jika wajah tidak terdeteksi di kondisi pencahayaan rendah |
| `Durasi Alarm` | Durasi minimum kantuk (detik) sebelum alarm aktif | Naikkan untuk mengurangi false alarm, turunkan untuk respons lebih cepat |

### Hardware Acceleration

Sistem secara otomatis mendeteksi ketersediaan GPU:
- **CUDA tersedia** → model berjalan di GPU (lebih cepat)
- **CUDA tidak tersedia** → fallback ke CPU (lebih lambat tapi tetap berjalan)

---

## ⚠️ Limitasi & Catatan

| Limitasi | Detail |
|---|---|
| **Browser Permission** | Browser harus mengizinkan akses kamera (popup permission) |
| **HTTPS Required** | WebRTC membutuhkan HTTPS atau localhost untuk akses kamera |
| **Pencahayaan** | Performa optimal pada kondisi pencahayaan yang cukup |
| **Jarak Kamera** | Wajah harus terlihat jelas dan cukup besar di frame kamera |
| **Single Face** | Dioptimasi untuk deteksi satu pengemudi (walaupun secara teknis bisa multi-face) |
| **Audio Autoplay** | Beberapa browser mungkin memblokir autoplay audio — interaksi pengguna (klik) dibutuhkan terlebih dahulu |

---

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan:

1. **Fork** repository ini
2. Buat **branch** fitur baru (`git checkout -b fitur-baru`)
3. **Commit** perubahan (`git commit -m 'Menambahkan fitur baru'`)
4. **Push** ke branch (`git push origin fitur-baru`)
5. Buat **Pull Request**

---

## 📜 Lisensi

Proyek ini dibuat sebagai **Proyek Akhir Machine Learning**.

---

<div align="center">

**Dibuat dengan ❤️ menggunakan Python, PyTorch, dan Streamlit**

*Mendeteksi kantuk, menyelamatkan nyawa.*

</div>