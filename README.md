# SirenMaster-TinyML 🚨

**SirenMaster-TinyML** adalah sistem deteksi suara sirine kendaraan darurat waktu-nyata (*real-time emergency vehicle siren detector*) berbasis Kecerdasan Buatan (AI) di perangkat tepi (*edge device*). Sistem ini dirancang menggunakan teknik **TinyML** dengan model **TensorFlow Lite Micro** untuk dijalankan di mikrokontroler **ESP32** dan **laptop/komputer**.

Sistem ini dapat mengenali 4 kategori suara:
1. **AMBULANCE** (Ambulans)
2. **FIRETRUCK** (Pemadam Kebakaran)
3. **POLICE** (Polisi)
4. **NORMAL** (Suara jalanan biasa, klakson, mesin, keheningan, dll.)

---

## 🌟 Fitur Utama

- **Real-Time DSP & Inference**: Pemrosesan sinyal digital (*Digital Signal Processing*) menggunakan Log-Mel Spectrogram (40 koefisien Mel, 249 frame waktu) yang sinkron 100% antara C++ (ESP32) dan Python (Laptop).
- **Arsitektur CNN 2D Ringan**: Dirancang khusus agar muat di memori SRAM internal ESP32 (alokasi memori hemat energi ~85 KB).
- **Active Segment Replication (Tiling)**: Menghindari distorsi jeda hening (*silence*) di awal sirine berbunyi pada mikrofon laptop.
- **Hot-Swap Online Learning**: Fitur koreksi pintar langsung dari keyboard laptop. Jika AI salah menebak, pengguna cukup menekan tombol angka (`1`-`4`) untuk merekam 8 detik suara terakhir, melakukan *fine-tuning* model lokal dalam waktu <15 detik, dan memperbarui model TFLite laptop serta kode C++ mikrokontroler (`model.h`) secara otomatis tanpa memutus aliran deteksi.
- **Microcontroller Integration**: Output fisik pada ESP32 berupa layar LCD grafis HUD futuristik, Strobo LED RGB interaktif, dan getaran taktil dari motor getar.

---

## 📂 Struktur File Proyek

- **`test_laptop.py`**: Script utama untuk menjalankan pendeteksian mikrofon secara realtime di laptop dengan visualisasi volume bar.
- **`fine_tune.py`**: Logika transfer learning cepat (*micro fine-tuning*) untuk fitur hot-swap online learning.
- **`train_lokal.py`**: Pipeline pelatihan penuh model dari awal dengan augmentasi data tingkat lanjut (*muffling, noise mixing, pitch shifting*).
- **`check_debug_mic.py`**: Alat bantu debug untuk mengevaluasi akurasi rekaman audio mentah.
- **`sirenmaster_main/`**: Folder kode program mikrokontroler ESP32 untuk Arduino IDE:
  - `sirenmaster_main.ino`: Program utama ESP32 menggunakan I2S audio (*Dual Core Task Scheduling*).
  - `model.h`: Berisi model TFLite terkuantisasi (INT8) dan parameter normalisasi Z-score.
  - `button_manager.cpp` / `buttonmanager.h`: Manajer tombol daya untuk Deep Sleep.

---

## 🔌 Skema Wiring Pin ESP32

Berikut adalah pinout koneksi perangkat keras pada mikrokontroler ESP32:

| Komponen | Pin Perangkat | Pin ESP32 | Keterangan |
| :--- | :--- | :--- | :--- |
| **I2S Microphone (INMP441)** | WS | **GPIO 32** | Word Select (I2S CLK) |
| | SCK | **GPIO 33** | Serial Clock |
| | SD | **GPIO 35** | Serial Data |
| | VCC / GND | **3.3V / GND** | Daya |
| **Vibration Motor** | Signal | **GPIO 13** | Output Motor Getar |
| **LED RGB** | Red / Green / Blue | **GPIO 26 / 27 / 21** | Output Strobo Warna |
| **LCD ST7789 (SPI)** | SCL (SCK) / SDA (MOSI) | **GPIO 18 / 23** | SPI Hardware |
| | RES (Reset) / DC | **GPIO 4 / 22** | Kontrol Display |
| | CS (Chip Select) / BL | **GPIO 14 / 15** | CS & Backlight PWM |
| **Push Button** | Signal | **GPIO 12** | Tombol Power (Deep Sleep) |

---

## 🚀 Cara Menjalankan Detektor di Laptop

1. **Instalasi Dependensi**:
   Pastikan Anda menggunakan Python 3.10 atau 3.11, lalu instal pustaka berikut:
   ```bash
   pip install numpy tensorflow sounddevice scipy librosa
   ```

2. **Jalankan Pendeteksian Real-Time**:
   Hubungkan mikrofon laptop Anda, lalu jalankan perintah:
   ```bash
   python test_laptop.py
   ```

3. **Hot-Swap Koreksi Kelas**:
   Saat script sedang mendengarkan, jika AI salah menebak suara sirine:
   - Tekan `1` jika suara tersebut seharusnya **AMBULANCE**
   - Tekan `2` jika suara tersebut seharusnya **FIRETRUCK**
   - Tekan `3` jika suara tersebut seharusnya **NORMAL**
   - Tekan `4` jika suara tersebut seharusnya **POLICE**
   
   Sistem akan merekam, melatih ulang model AI, dan mengonversinya kembali menjadi C++ `model.h` secara instan.
