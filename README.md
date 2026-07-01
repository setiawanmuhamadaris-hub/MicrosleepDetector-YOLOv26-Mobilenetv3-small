# MicrosleepDetector-YOLOv26-Mobilenetv3-small

Sistem Deteksi Microsleep Real-Time menggunakan YOLOv26 (Deteksi Wajah) dan MobileNetV3 (Klasifikasi Kantuk). Aplikasi ini berjalan di atas Streamlit dan mendeteksi kantuk melalui kamera perangkat secara real-time.

## Cara Menjalankan Aplikasi Secara Lokal (Windows)

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi di lingkungan lokal Anda.

### 1. Prasyarat
Pastikan Anda sudah menginstal **Python** (disarankan versi 3.9 - 3.11).

### 2. Mengaktifkan Virtual Environment
Jika virtual environment `env-mocrosleep` sudah tersedia, Anda bisa langsung mengaktifkannya melalui PowerShell atau Command Prompt.

Buka terminal pada direktori proyek ini lalu jalankan:

```powershell
.\env-mocrosleep\Scripts\activate
```

*(Catatan: Jika virtual environment belum ada, Anda bisa membuatnya terlebih dahulu dengan `python -m venv env-mocrosleep`)*

### 3. Instalasi Library (Jika belum terinstal)
Pastikan semua dependensi terinstal. Jalankan perintah berikut di dalam virtual environment:

```powershell
pip install streamlit opencv-python torch torchvision ultralytics numpy
```

### 4. Menjalankan Aplikasi
Setelah virtual environment aktif dan semua library terinstal, jalankan aplikasi Streamlit dengan perintah:

```powershell
streamlit run app.py
```

Perintah di atas akan secara otomatis membuka aplikasi di browser Anda (biasanya di `http://localhost:8501`).

### 5. Penggunaan
- Pada antarmuka aplikasi, centang opsi **"Mulai Kamera 🎥"** di bagian sidebar.
- Pastikan wajah Anda terlihat jelas oleh kamera web.
- Jika terdeteksi "Microsleep", sistem akan memberikan peringatan visual dan membunyikan alarm.