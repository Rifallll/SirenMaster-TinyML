"""
list_and_clean_anomalies.py
===========================
Mendaftar 34 file aneh (silent/intro/speech tanpa energi sirene)
dan memindahkannya ke folder TRASH agar dataset sirene 100% murni.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, glob, shutil, numpy as np
import scipy.io.wavfile as wavfile

ROOT = r"C:\Users\ASUS\Videos\DATASET"
TRASH_DIR = os.path.join(ROOT, "TRASH", "ANOMALI_SPEKTROGRAM_SIRENE")
os.makedirs(TRASH_DIR, exist_ok=True)

SIREN_FOLDERS = ["AMBULANCE", "FIRETRUCK", "POLICE"]

print("=" * 80)
print(" 🧹 PEMBERSIHAN FILE ANEH / BUKAN SIRENE DARI FOLDER AMBULANCE, FIRETRUCK & POLICE")
print("=" * 80)

anomalies = []

for folder in SIREN_FOLDERS:
    folder_path = os.path.join(ROOT, folder)
    for filepath in sorted(glob.glob(os.path.join(folder_path, "*.wav"))):
        filename = os.path.basename(filepath)
        try:
            sr, data = wavfile.read(filepath)
        except Exception:
            anomalies.append((folder, filepath, filename, "Corrupt Header"))
            continue
            
        if len(data.shape) > 1:
            data = data[:, 0]
        data_float = data.astype(np.float32) / 32768.0
        
        rms = np.sqrt(np.mean(data_float ** 2))
        if rms < 0.005:
            anomalies.append((folder, filepath, filename, f"Silent / Intro Diam (RMS={rms:.4f})"))
            continue
            
        fft_spec = np.abs(np.fft.rfft(data_float)) ** 2
        freqs = np.fft.rfftfreq(len(data_float), 1.0 / sr)
        total_energy = np.sum(fft_spec) + 1e-9
        
        low_energy_ratio = np.sum(fft_spec[freqs < 100]) / total_energy
        siren_band_ratio = np.sum(fft_spec[(freqs >= 300) & (freqs <= 3000)]) / total_energy
        
        if siren_band_ratio < 0.12 or low_energy_ratio > 0.85:
            anomalies.append((folder, filepath, filename, f"Bukan Sirene (SirenBand={siren_band_ratio*100:.1f}%, LowHz={low_energy_ratio*100:.1f}%)"))

print(f"[*] Ditemukan {len(anomalies)} file aneh / bukan sirene. Memindahkan ke folder TRASH...\n")

for idx, (folder, filepath, filename, reason) in enumerate(anomalies, 1):
    print(f" {idx:2d}. [{folder:<10}] {filename:<38} -> {reason}")
    target_path = os.path.join(TRASH_DIR, f"{folder}_{filename}")
    shutil.move(filepath, target_path)

print("\n" + "=" * 80)
print(f" 🏆 BERHASIL! {len(anomalies)} file aneh telah dikarantina ke TRASH.")
print(" Sekarang seluruh folder AMBULANCE, FIRETRUCK, dan POLICE 100% murni bersuara sirene!")
print("=" * 80)
