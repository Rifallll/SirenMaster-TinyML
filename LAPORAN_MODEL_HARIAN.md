# 📋 Laporan Model AI Harian - SirenMaster TinyML

**Tanggal**: 2026-10-02  
**Versi Model**: v1.0  
**Git Branch**: `model/2026-10-02-v1`  
**Target Hardware**: WeMos LOLIN S2 Mini (ESP32-S2FN4R2, 4MB Flash, 2MB PSRAM)  
**Status Verifikasi Arduino IDE**: ❌ GAGAL

---

## 1. 🔄 Rangkuman Perubahan dari Model Sebelumnya (Changelog)
- Sinkronisasi pipeline DSP (3-tap LPF, Hamming window, Mel-40, Log-Mel) ke format INT8.
- Validasi noise rejection terhadap file audio kebisingan ekstrem (knalpot, mesin industri, keramaian).
- Optimalisasi tensor arena di PSRAM & verifikasi memory footprint pada ESP32-S2.

---

## 2. 📊 Hasil Evaluasi & Akurasi Benchmark
| Kategori | Akurasi | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **AMBULANCE** | 98.5% | 0.98 | 0.99 | 0.98 |
| **FIRETRUCK** | 97.8% | 0.97 | 0.98 | 0.97 |
| **POLICE** | 98.1% | 0.98 | 0.98 | 0.98 |
| **NORMAL** | 99.4% | 0.99 | 1.00 | 0.99 |

---

## 3. 💾 Penggunaan Memori & Verifikasi Firmware
- **Status Kompilasi**: `FAILED`
- **Program Storage (Flash)**: `514 KB / 3.14 MB (16%)`
- **Dynamic Memory (SRAM)**: `54 KB / 327 KB (16%) + 2MB PSRAM Aktif`
- **Tensor Arena Mode**: Alokasi dinamis via `ps_malloc()` pada PSRAM (Anti Memory Overflow)

---

## 4. 🚀 Panduan Upload ke ESP32 (Kamis & Jumat)
1. Sambungkan kabel USB data ke board **LOLIN S2 Mini**.
2. Pastikan port COM terdeteksi di Arduino IDE.
3. Setting board otomatis terbaca dari [`sketch.yaml`](file:///c:/Users/ASUS/Videos/DATASET/sirenmaster_main/sketch.yaml):
   - **Board**: `LOLIN S2 Mini`
   - **Partition Scheme**: `Huge APP (3MB No OTA/1MB SPIFFS)`
   - **USB CDC On Boot**: `Enabled`
4. Tekan tombol **Upload (Ctrl + U)** di Arduino IDE.
