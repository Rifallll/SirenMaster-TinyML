"""
plot_real_spectrograms.py
=========================
Script untuk menghasilkan gambar grafik Mel-Spectrogram nyata (249x40) dari dataset kita
untuk memperlihatkan perbedaan visual antara Ambulance, Police, Firetruck, dan Normal.
"""
import os
import glob
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Path folder dataset
DATASET_DIR = r"C:\Users\ASUS\Videos\DATASET"
OUT_IMG_ARTIFACT = r"C:\Users\ASUS\.gemini\antigravity-ide\brain\476d7d9e-a9aa-4fef-8a18-4423510e95dd\siren_spectrogram_comparison.png"
OUT_IMG_LOCAL = r"C:\Users\ASUS\Videos\DATASET\04_Training_AI\siren_spectrogram_comparison.png"

def get_sample_file(folder_name, keyword=""):
    folder_path = os.path.join(DATASET_DIR, folder_name)
    files = glob.glob(os.path.join(folder_path, "*.wav")) + glob.glob(os.path.join(folder_path, "*.mp3"))
    if keyword:
        kw_files = [f for f in files if keyword.lower() in os.path.basename(f).lower()]
        if kw_files:
            return kw_files[0]
    return files[0] if files else None

def compute_melspec(file_path):
    y, sr = librosa.load(file_path, sr=8000, duration=4.0)
    # Pad jika kurang dari 4 detik
    if len(y) < 32000:
        y = np.pad(y, (0, 32000 - len(y)))
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=256, hop_length=128, n_mels=40, fmax=4000
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel, sr

print("[*] Mengambil sampel audio asli dari dataset...")
f_amb = get_sample_file("AMBULANCE")
f_pol = get_sample_file("POLICE")
f_fir = get_sample_file("FIRETRUCK")
f_nor = get_sample_file("NORMAL", keyword="knalpot") or get_sample_file("NORMAL")

samples = [
    ("AMBULANCE (Gelombang Bukit Lambat)", f_amb, "Blues"),
    ("POLICE (Gigi Gergaji Rapat & Cepat)", f_pol, "Reds"),
    ("FIRETRUCK (Raungan Bass Berat di Bawah)", f_fir, "Oranges"),
    ("NORMAL - Knalpot/Jalanan (Datar & Acak)", f_nor, "Greens")
]

plt.figure(figsize=(14, 10))
plt.suptitle("BUKTI REAL: Cara AI Membedakan Suara Sirine vs Jalanan (Mel-Spectrogram 249x40)", fontsize=16, fontweight='bold', y=0.98)

for i, (title, fpath, cmap) in enumerate(samples, 1):
    plt.subplot(2, 2, i)
    if fpath and os.path.exists(fpath):
        log_mel, sr = compute_melspec(fpath)
        img = librosa.display.specshow(
            log_mel, sr=sr, hop_length=128, x_axis='time', y_axis='mel', fmax=4000, cmap='magma'
        )
        plt.colorbar(format='%+2.0f dB')
        fname = os.path.basename(fpath)
        plt.title(f"{title}\nFile: {fname[:30]}...", fontsize=11, fontweight='bold', pad=10)
        plt.xlabel("Waktu (Detik) -> 249 Bingkai", fontsize=10)
        plt.ylabel("Frekuensi Mel (Hz) -> 40 Kolom", fontsize=10)
    else:
        plt.title(f"{title} (File tidak ditemukan)", fontsize=11)

plt.tight_layout()
plt.subplots_adjust(top=0.90)

# Simpan gambar
os.makedirs(os.path.dirname(OUT_IMG_ARTIFACT), exist_ok=True)
plt.savefig(OUT_IMG_ARTIFACT, dpi=150)
plt.savefig(OUT_IMG_LOCAL, dpi=150)
plt.close()

print(f"[SUCCESS] Gambar berhasil disimpan di:\n  -> {OUT_IMG_ARTIFACT}\n  -> {OUT_IMG_LOCAL}")
