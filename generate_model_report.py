import os
import sys
import datetime

def create_report(
    output_md_path="LAPORAN_MODEL_HARIAN.md"
):
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    branch_name = f"model/{today_str}-v1"

    report_content = f"""# 📋 Laporan Evaluasi & Bedah Model AI SirenMaster TinyML

**Tanggal**: {today_str}  
**Git Branch**: `{branch_name}`  
**Target Hardware**: WeMos LOLIN S2 Mini (ESP32-S2FN4R2, 4MB Flash, 2MB PSRAM)  
**Status Verifikasi Arduino IDE**: ✅ PASSED (0 Error, Flash: 16%, SRAM: 16%)  
**Dokumen Word**: [`LAPORAN_MODEL_HARIAN.docx`](file:///c:/Users/ASUS/Videos/DATASET/LAPORAN_MODEL_HARIAN.docx)  
**Data CSV Evaluasi**: [`evaluation_report.csv`](file:///c:/Users/ASUS/Videos/DATASET/evaluation_report.csv)  

---

## 1. 📊 Hasil Evaluasi Dataset Validasi Riil (2.764 Sampel Audio)
Sumber data: [`evaluation_report.csv`](file:///c:/Users/ASUS/Videos/DATASET/evaluation_report.csv)

| Kategori Suara | Precision | Recall | F1-Score | Jumlah Sampel (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **AMBULANCE** | 64.5% | 82.8% | **72.5%** | 145 file |
| **FIRETRUCK (Damkar)** | 93.9% | 66.7% | **78.0%** | 1.328 file |
| **POLICE (Polisi)** | 68.7% | 91.6% | **78.5%** | 939 file |
| **NORMAL (Kebisingan)** | 84.1% | 91.5% | **87.6%** | 352 file |
| **RATA-RATA TOTAL (Accuracy)** | **79.2%** | **79.2%** | **79.2%** | **2.764 file** |

---

## 2. 🧪 Hasil Pengujian Live Hardware 100 Sirine
Sumber data: [`hasil_benchmark_100.txt`](file:///c:/Users/ASUS/Videos/DATASET/hasil_benchmark_100.txt)

* **Akurasi Total Live**: **57/100 (57.0%)**
* **Berhasil Terdeteksi Benar**: 57 audio
* **Diam (Tidak Memicu Siren)**: 32 audio
* **Salah Kamar (Tertukar Kelas)**: 11 audio

### Rincian Performa Live Per Kelas:
- **AMBULANCE**: 31/33 (**93.9%**) — *Sangat Responsif*
- **DAMKAR (FIRETRUCK)**: 4/33 (**12.1%**) — *Penyumbang error terbesar (banyak salah kamar/diam)*
- **POLISI (POLICE)**: 22/34 (**64.7%**) — *Cukup Baik*

---

## 3. 🔍 Analisis Penyebab & Solusi

### A. Mengapa Ada 32 Suara "DIAM" (Tidak Memicu Siren)?
- **Penyebab**: Syarat pemicu sirene pada C++ ESP32 diset ketat (`bestSirenScore >= 0.58f && cs[bestSiren] >= 0.60f && bestSirenScore > cs[2]`). Jika suara sirene diputar dengan volume rendah atau dari jarak jauh, skor sirine hanya mencapai 35%-50% sehingga firmware menahannya di kelas Normal agar tidak terjadi false alarm di jalan raya.

### B. Mengapa Terjadi 11 Suara "SALAH KAMAR" (Terutama Damkar 12.1%)?
- **Penyebab**: Nada *Yelp* Damkar memiliki frekuensi yang beririsan dengan Polisi dan nada *Hi-Lo* beririsan dengan Ambulans. Karena data latih Polisi/Ambulans lebih banyak, model AI cenderung condong memilih Polisi atau Ambulans ketika ragu.
- **Solusi**: Menjalankan [`fix_firetruck_augment.py`](file:///c:/Users/ASUS/Videos/DATASET/fix_firetruck_augment.py) untuk memperkaya dataset Damkar dengan variasi pitch shift, speed stretch, dan noise mixing.

---

## 4. 💾 Hasil Verifikasi Memori Arduino IDE
- **Status Kompilasi**: `PASSED`
- **Program Storage (Flash)**: `514.471 bytes (16%)` dari maks `3.145.728 bytes`
- **Dynamic Memory (SRAM)**: `54.352 bytes (16%)` dari maks `327.680 bytes`
- **External PSRAM**: Aktif (`ps_malloc` untuk Tensor Arena)
"""

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[+] File markdown laporan riil berhasil diperbarui: {output_md_path}")
    return report_content

if __name__ == "__main__":
    create_report()
