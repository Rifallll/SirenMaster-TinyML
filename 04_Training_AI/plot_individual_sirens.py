"""
plot_individual_sirens.py
=========================
Menghasilkan 4 gambar terpisah resolusi tinggi (Waveform + Mel-Spectrogram) untuk
Ambulance, Police, Firetruck, dan Normal agar perbedaan masing-masing sangat jelas!
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

def plot_single_class(folder_name, title_id, desc, cmap, out_name, keyword=""):
    fpath = get_sample_file(folder_name, keyword)
    if not fpath or not os.path.exists(fpath):
        print(f"[!] File untuk {folder_name} tidak ditemukan.")
        return

    print(f"[*] Memproses {folder_name}: {os.path.basename(fpath)} ...")
    y, sr = librosa.load(fpath, sr=8000, duration=4.0)
    if len(y) < 32000:
        y = np.pad(y, (0, 32000 - len(y)))
        
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=256, hop_length=128, n_mels=40, fmax=4000
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    time_axis = np.linspace(0, 4.0, len(y))

    plt.figure(figsize=(12, 8))
    
    # 1. Waveform (Gelombang Suara Mentah)
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, y, color='darkblue', alpha=0.8)
    plt.title(f"1. Gelombang Suara Mentah (Waveform) - {title_id}\nFile: {os.path.basename(fpath)}", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Waktu (Detik)", fontsize=11)
    plt.ylabel("Amplitudo / Keras Suara", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 4.0)
    
    # 2. Mel-Spectrogram (Yang Dilihat AI ESP32)
    plt.subplot(2, 1, 2)
    img = librosa.display.specshow(
        log_mel, sr=sr, hop_length=128, x_axis='time', y_axis='mel', fmax=4000, cmap=cmap
    )
    cbar = plt.colorbar(format='%+2.0f dB')
    cbar.set_label('Energi Suara (Desibel)', fontsize=10)
    plt.title(f"2. Grafik Mel-Spectrogram 249x40 (Yang Dilihat AI ESP32)\nCiri Khas: {desc}", fontsize=12, fontweight='bold', pad=10, color='darkred')
    plt.xlabel("Waktu -> 249 Bingkai (0,016s / bingkai)", fontsize=11)
    plt.ylabel("Frekuensi Mel (Hz) -> 40 Kolom", fontsize=11)
    
    plt.tight_layout()
    
    art_path = os.path.join(ARTIFACT_DIR, out_name)
    loc_path = os.path.join(LOCAL_DIR, out_name)
    plt.savefig(art_path, dpi=150)
    plt.savefig(loc_path, dpi=150)
    plt.close()
    print(f"    [OK] Tersimpan: {out_name}")

print("=" * 60)
print("  MEMBUAT GAMBAR GRAFIK INDIVIDUAL RESOLUSI TINGGI")
print("=" * 60)

plot_single_class(
    "AMBULANCE", "AMBULANCE (Ambulans)", 
    "Gelombang bukit melengkung mulus & naik-turun lambat (2-3 detik per siklus) di frekuensi tengah.",
    "magma", "spectrogram_ambulance.png"
)

plot_single_class(
    "POLICE", "POLICE (Polisi - Yelp)", 
    "Deretan gigi gergaji rapat & curam! Melompat cepat (0,4-0,5 detik per siklus) ke nada tinggi.",
    "inferno", "spectrogram_police.png"
)

plot_single_class(
    "FIRETRUCK", "FIRETRUCK (Pemadam Kebakaran)", 
    "Pita energi tebal & konstan di bagian bawah (Nada Bass Rendah 200-800 Hz) tanpa putus.",
    "plasma", "spectrogram_firetruck.png"
)

plot_single_class(
    "NORMAL", "NORMAL (Berisik Jalanan / Knalpot Brong)", 
    "Pita mendatar lurus atau acak tanpa pola gelombang periodik! AI langsung mengenali ini bukan sirine.",
    "viridis", "spectrogram_normal.png", keyword="knalpot"
)

print("\n[SUCCESS] Semua gambar individual berhasil dibuat!")
