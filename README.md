# SirenMaster-TinyML 🚨

**SirenMaster-TinyML** adalah sistem deteksi suara sirine kendaraan darurat waktu-nyata (*real-time emergency vehicle siren detector*) berbasis Kecerdasan Buatan (AI) di perangkat tepi (*edge device*). Sistem ini dirancang menggunakan teknik **TinyML** dengan model **TensorFlow Lite Micro (Full INT8 Quantization)** untuk dijalankan pada mikrokontroler **ESP32-S3 / ESP32 WROOM** dan **Laptop/Komputer**.

Sistem ini mengenali 4 kategori suara secara presisi:
1. 🚑 **AMBULANCE** (Ambulans — Nada *Wail / Yelp / Hi-Lo*)
2. 🚒 **FIRETRUCK** (Pemadam Kebakaran — Nada *Powercall / Horn / Yelp*)
3. 🚓 **POLICE** (Polisi — Nada *Phaser / Fast Yelp / Wail*)
4. 🛡️ **NORMAL** (Suara jalanan biasa, bising mesin, percakapan, klakson, telolet, musik, keheningan)

---

## 📊 Performa & Hasil Evaluasi Model AI (Test Set 1.004 Sampel)

Model dilatih menggunakan arsitektur **2D CNN (Separable Convolutional Network)** dengan ukuran terkuantisasi hanya **20.09 KB** sehingga muat di memori SRAM internal ESP32:

- **Akurasi Keseluruhan (Test Accuracy):** **95.1%**
- **Macro F1-Score:** **0.95**
- **Normal Noise Rejection:** **99.9% F1-Score** (Nihil False Alarm pada kebisingan umum)

| Kategori | Precision | Recall | F1-Score | Jumlah Test Set |
| :--- | :---: | :---: | :---: | :---: |
| 🚑 **AMBULANCE** | **95%** | **91%** | **0.93** | 256 sampel |
| 🚒 **FIRETRUCK** | **93%** | **97%** | **0.95** | 242 sampel |
| 🚓 **POLICE** | **92%** | **93%** | **0.93** | 218 sampel |
| 🛡️ **NORMAL** | **100%** | **98%** | **0.99** | 288 sampel |

---

## 🌟 Fitur Utama & Inovasi Teknologi

- **Temporal Consensus Hard Siren Locking**: Penguncian kelas mutlak yang mencegah *flip-flopping* atau lompatan status di tengah-tengah nada sirine akibat resonansi/gema ruangan atau penurunan volume ekor nada.
- **Auto-Gain Match (10.0x Peak Normalization)**: Sinkronisasi gain dinamik 100% identik antara pemrosesan DSP C++ pada ESP32 dan ekstraksi fitur Python pada laptop.
- **Ultra-Fast Initial Trigger**: Respon secepat kilat (mengunci sejak Putaran 1-2 pemutaran audio) pada probabilitas sirine $\ge 30\%$.
- **DSP Feature Extraction**: Log-Mel Spectrogram (40 Mel Filters, 249 Time Frames, Hop 128, FFT 256, Hamming Window, Frame DC Offset Removal).
- **Dual-Core FreeRTOS Task Scheduling**: Pemisahan tugas pembacaan I2S audio (`TaskAudio` di Core 1) dan inferensi AI/LCD HUD UI (`TaskInference` di Core 0).
- **Cinematic TFT LCD & Audio Tester Utility**: Visualisasi HUD interaktif pada ST7789 dan script pengujian otomatis `putar_sirene.py`.

---

## 📂 Struktur Repositori Proyek

```text
SirenMaster-TinyML/
├── 📁 sirenmaster_main/          # Codebase Firmware ESP32 (Arduino IDE)
│   ├── sirenmaster_main.ino      # Program utama C++ ESP32 (FreeRTOS + TFLite Micro)
│   ├── model.h                   # Model INT8 TFLite C++ Deployment Header
│   ├── siren_model_data.h        # Data TFLite Flatbuffer Array
│   ├── button_manager.cpp        # Manajer Tombol Power & Deep Sleep
│   └── buttonmanager.h
├── 📁 04_Training_AI/            # Script Pelatihan & Evaluasi Model
│   ├── train_lokal.py            # Pipeline retraining model dari awal
│   └── test_prediction.py        # Pengujian inferensi model lokal
├── 📁 AMBULANCE/                 # Dataset Murni Ambulans (1.513 file)
├── 📁 FIRETRUCK/                 # Dataset Murni Damkar (2.048 file)
├── 📁 POLICE/                    # Dataset Murni Polisi (1.360 file)
├── 📁 NORMAL/                    # Dataset Murni Kebisingan Jalanan (9.383 file)
├── 📄 putar_sirene.py            # Script penguji otomatis via speaker laptop
├── 📄 siren_model_quant.tflite   # Binary model TFLite INT8 (20 KB)
├── 📄 siren_scaler.npz           # Parameter Z-Score Mean & Std
└── 📄 README.md                  # Dokumentasi Proyek
```

---

## 🔌 Skema Wiring Pin ESP32-S3 / ESP32 WROOM

| Perangkat / Komponen | Pin Perangkat | Pin ESP32 / ESP32-S3 | Keterangan |
| :--- | :--- | :--- | :--- |
| **I2S Microphone (INMP441)** | WS | **GPIO 5** | Word Select (I2S Clock) |
| | SCK | **GPIO 6** | Serial Clock |
| | SD | **GPIO 7** | Serial Data |
| | VCC / GND | **3.3V / GND** | Catu Daya |
| **Vibration Motor** | Signal | **GPIO 13** | Output Getaran Taktil |
| **Strobo LED RGB** | Red / Green / Blue | **GPIO 38 / 39 / 40** | Output Indikator Warna |
| **ST7789 TFT LCD Display** | SCL / SDA | **GPIO 36 / 35** | SPI Hardware |
| | RES / DC | **GPIO 18 / 16** | Reset & Data/Command |
| | CS / BL | **GPIO 10 / 9** | Chip Select & Backlight PWM |
| **Push Button** | Signal | **GPIO 8** | Mute / Snooze 10 Detik |

---

## 🚀 Cara Menjalankan & Pengujian Proyek

### 1. Pengujian Audio di Laptop (`putar_sirene.py`)
Jalankan penguji audio otomatis untuk membunyikan sirine acak melalui speaker laptop ke mikrokontroler:
```bash
python putar_sirene.py
```
Pilih opsi **[1] Ujian 10 Sirine Random** atau **[4] Benchmark 100 Sirine**.

### 2. Kompilasi & Deploy ke ESP32 (Arduino IDE)
1. Buka folder `sirenmaster_main/` di **Arduino IDE 2.x**.
2. Pilih Board: **ESP32S3 Dev Module** (atau **ESP32 Dev Module**).
3. Pilih Partition Scheme: **Huge APP (3MB No OTA / 1MB SPIFFS)**.
4. Hubungkan ESP32 via kabel USB, pilih port COM yang sesuai (misal: `COM6`).
5. Klik **Upload (➔)** untuk mengompilasi dan mengunggah program ke ESP32.

---

## 📜 Lisensi & Kontribusi
Dikembangkan oleh **Rifallll** untuk proyek **SirenMaster TinyML Emergency Vehicle Detection**. Open-source di bawah lisensi MIT.
