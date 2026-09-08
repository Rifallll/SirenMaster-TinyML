"""
plot_skripsi_figure.py
======================
Script khusus untuk menghasilkan 1 GAMBAR AKADEMIK SKRIPSI (4 Baris x 2 Kolom) resolusi tinggi
yang menggabungkan keempat kelas (Ambulance, Police, Firetruck, Normal) dalam satu gambar,
lengkap dengan label frekuensi (Hz), waktu (detik), dan penjelasan parameter ilmiahnya!
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import glob
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = r"C:\Users\ASUS\Videos\DATASET"
ARTIFACT_DIR = r"C:\Users\ASUS\.gemini\antigravity-ide\brain\476d7d9e-a9aa-4fef-8a18-4423510e95dd"
LOCAL_DIR = r"C:\Users\ASUS\Videos\DATASET\04_Training_AI"

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(LOCAL_DIR, exist_ok=True)

def get_sample_file(folder_name, keyword=""):
    folder_path = os.path.join(DATASET_DIR, folder_name)
    files = glob.glob(os.path.join(folder_path, "*.wav")) + glob.glob(os.path.join(folder_path, "*.mp3"))
    if keyword:
        kw_files = [f for f in files if keyword.lower() in os.path.basename(f).lower()]
        if kw_files:
            return kw_files[0]
    return files[0] if files else None

print("[*] Mengambil sampel audio untuk gambar skripsi...")
f_amb = get_sample_file("AMBULANCE")
f_pol = get_sample_file("POLICE")
f_fir = get_sample_file("FIRETRUCK")
f_nor = get_sample_file("NORMAL", keyword="knalpot") or get_sample_file("NORMAL")

classes_data = [
    (
        "AMBULANCE (Ambulans)", f_amb, "magma",
        "Dominan: 500 - 1.500 Hz | Siklus: 2,0 - 3,0 detik (Slow Wave / Gelombang Bukit Lambat)"
    ),
    (
        "POLICE (Polisi - Yelp)", f_pol, "inferno",
        "Dominan: 600 - 2.500+ Hz | Siklus: 0,4 - 0,5 detik (Fast Sawtooth / Gigi Gergaji Rapat)"
    ),
    (
        "FIRETRUCK (Damkar)", f_fir, "plasma",
        "Dominan: 200 - 800 Hz (Heavy Bass) | Siklus: Konstan > 2,0 detik (Rumbler / Q-Siren)"
    ),
    (
        "NORMAL (Jalanan / Knalpot)", f_nor, "viridis",
        "Spektrum: Pita Datar / Acak | Tidak Ada Siklus Periodik (Anti-False Alarm)"
    )
]

# Buat Figure 4 Baris x 2 Kolom (Resolusi Tinggi untuk Skripsi)
plt.figure(figsize=(16, 16))
plt.suptitle("ANALISIS KARAKTERISTIK AKUSTIK DAN MEL-SPECTROGRAM DATASET SIRENMASTER (4 KELAS)\n(Bukti Empiris untuk Klasifikasi Audio TinyML pada Board ESP32)", fontsize=16, fontweight='bold', y=0.98)

for i, (name, fpath, cmap, desc) in enumerate(classes_data):
    row_idx = i * 2
    
    if not fpath or not os.path.exists(fpath):
        print(f"[!] File untuk {name} tidak ditemukan.")
        continue
        
    y, sr = librosa.load(fpath, sr=8000, duration=4.0)
    if len(y) < 32000:
        y = np.pad(y, (0, 32000 - len(y)))
        
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=256, hop_length=128, n_mels=40, fmax=4000
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    time_axis = np.linspace(0, 4.0, len(y))
    
    # 1. Kolom Kiri: Waveform (Gelombang Amplitudo vs Waktu)
    plt.subplot(4, 2, row_idx + 1)
    plt.plot(time_axis, y, color='#003366', alpha=0.85, linewidth=0.8)
    plt.title(f"({chr(97+row_idx)}) Waveform {name}\n[{desc}]", fontsize=11, fontweight='bold', pad=6, color='#1a1a1a')
    plt.ylabel("Amplitudo", fontsize=10)
    if i == 3:
        plt.xlabel("Waktu (Detik)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 4.0)
    plt.ylim(-1.0, 1.0)
    
    # 2. Kolom Kanan: Mel-Spectrogram (Frekuensi Hz vs Waktu)
    plt.subplot(4, 2, row_idx + 2)
    img = librosa.display.specshow(
        log_mel, sr=sr, hop_length=128, x_axis='time', y_axis='mel', fmax=4000, cmap=cmap
    )
    cbar = plt.colorbar(format='%+2.0f dB', pad=0.02)
    cbar.set_label('Energi (dB)', fontsize=9)
    plt.title(f"({chr(97+row_idx+1)}) Mel-Spectrogram {name} (249x40 Matrix)\n[Input Real Neural Network ESP32]", fontsize=11, fontweight='bold', pad=6, color='#800000')
    plt.ylabel("Frekuensi (Hz)", fontsize=10)
    if i == 3:
        plt.xlabel("Waktu (Detik) -> 249 Bingkai", fontsize=10)

plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.35, wspace=0.15)

out_art = os.path.join(ARTIFACT_DIR, "gambar_skripsi_all_classes.png")
out_loc = os.path.join(LOCAL_DIR, "gambar_skripsi_all_classes.png")

plt.savefig(out_art, dpi=200)
plt.savefig(out_loc, dpi=200)
plt.close()

print(f"[SUCCESS] Gambar skripsi resolusi tinggi berhasil dibuat:\n  -> {out_art}\n  -> {out_loc}")
