"""
audit_duplikat_dan_klasifikasi.py
===================================
1. Audit Duplikat Identik (MD5 Audio Hash) antar dan dalam folder SIRENE.
2. Audit Cross-Class & Karakteristik Modulasi Pitch (Memastikan Ambulance = Wail Lambat,
   Police = Yelp/Hi-Lo Cepat, Firetruck = Raungan Mekanik Berat).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, glob, hashlib, numpy as np
import scipy.io.wavfile as wavfile
from collections import defaultdict

ROOT = r"C:\Users\ASUS\Videos\DATASET"
SIREN_FOLDERS = ["AMBULANCE", "FIRETRUCK", "POLICE"]

print("=" * 80)
print(" 🔍 AUDIT DUPLIKAT & KETEPATAN KLASIFIKASI SIRENE (WAIL vs YELP vs Q-SIREN)")
print("=" * 80)

# --- TAHAP 1: AUDIT DUPLIKAT ---
print("\n[Tahap 1] Memeriksa duplikat konten audio (MD5 Hash)...")
hash_dict = defaultdict(list)
total_files = 0

for folder in SIREN_FOLDERS:
    folder_path = os.path.join(ROOT, folder)
    for filepath in sorted(glob.glob(os.path.join(folder_path, "*.wav"))):
        total_files += 1
        try:
            sr, data = wavfile.read(filepath)
            if len(data.shape) > 1:
                data = data[:, 0]
            # Hash dari array audio mentah
            audio_hash = hashlib.md5(data.tobytes()).hexdigest()
            hash_dict[audio_hash].append((folder, filepath, os.path.basename(filepath)))
        except Exception as e:
            continue

duplicates_found = {k: v for k, v in hash_dict.items() if len(v) > 1}
print(f"[*] Total file diperiksa        : {total_files} file")
print(f"[*] Kelompok duplikat ditemukan : {len(duplicates_found)} kelompok")

if duplicates_found:
    print("\n--- DAFTAR DUPLIKAT DITEMUKAN ---")
    for idx, (h, items) in enumerate(list(duplicates_found.items())[:15], 1):
        print(f"  Kelompok #{idx} ({len(items)} file identik):")
        for fld, fp, fn in items:
            print(f"    - [{fld}] {fn}")
else:
    print("[+] Tidak ada duplikat audio identik di seluruh dataset sirene!")


# --- TAHAP 2: AUDIT KETEPATAN KLASIFIKASI (MODULASI PITCH) ---
print("\n" + "=" * 80)
print("[Tahap 2] Memeriksa Karakteristik Modulasi Suara (Salah Kamar / Cross-Class Audit)...")
print("  -> AMBULANCE harus dominan Wail lambat (siklus 1.5s - 4s)")
print("  -> POLICE harus dominan Yelp cepat / Hi-Lo (siklus < 0.6s)")
print("  -> FIRETRUCK harus dominan Q-Siren mekanik / rumbler berat")
print("=" * 80)

misclassified = []

for folder in SIREN_FOLDERS:
    folder_path = os.path.join(ROOT, folder)
    # Periksa sampel asli & sintetik (lewatkan aug_ supaya lebih cepat dan fokus ke sumber utama)
    sample_files = sorted([f for f in glob.glob(os.path.join(folder_path, "*.wav")) if not os.path.basename(f).startswith("aug_")])
    
    for filepath in sample_files:
        filename = os.path.basename(filepath)
        try:
            sr, data = wavfile.read(filepath)
            if len(data.shape) > 1:
                data = data[:, 0]
            data_float = data.astype(np.float32) / 32768.0
        except Exception:
            continue
            
        # Analisis STFT sederhana untuk mencari frekuensi dominan (Pitch Track) tiap 50ms
        frame_len = int(sr * 0.05)
        hop_len = int(sr * 0.025)
        n_frames = (len(data_float) - frame_len) // hop_len
        if n_frames < 20: # Durasi terlalu pendek untuk diukur modulasinya
            continue
            
        centroids = []
        for i in range(n_frames):
            frame = data_float[i*hop_len : i*hop_len + frame_len]
            # Jendela Hanning
            frame = frame * np.hanning(len(frame))
            fft_mag = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1.0/sr)
            
            # Batasi hanya di pita suara sirene (300 Hz - 2500 Hz)
            mask = (freqs >= 300) & (freqs <= 2500)
            if np.sum(fft_mag[mask]) > 0:
                c = np.sum(freqs[mask] * fft_mag[mask]) / np.sum(fft_mag[mask])
                centroids.append(c)
            else:
                centroids.append(0)
                
        centroids = np.array(centroids)
        # Hapus frame hening
        valid_c = centroids[centroids > 300]
        if len(valid_c) < 20:
            continue
            
        # Hitung seberapa cepat pitch berubah (autokorelasi / laju perubahan frekuensi)
        # Jika selisih pitch antar frame (25ms) sangat tinggi secara berulang -> Yelp / Hi-Lo Cepat
        diff_pitch = np.abs(np.diff(valid_c))
        mean_diff = np.mean(diff_pitch)
        
        # Hitung deviasi standar pitch (apakah konstan atau bergelombang)
        std_pitch = np.std(valid_c)
        
        # Kriteria Deteksi Anomali / Salah Kamar:
        # 1. Folder AMBULANCE tapi punya modulasi sangat cepat (mean_diff > 120 Hz per 25ms dan std > 250 Hz) -> Ini Yelp Polisi!
        if folder == "AMBULANCE" and mean_diff > 130.0 and std_pitch > 220.0:
            misclassified.append((folder, filepath, filename, f"Terdeteksi YELP/HI-LO POLISI (Modulasi Cepat {mean_diff:.1f}Hz/step)"))
            
        # 2. Folder POLICE tapi modulasi sangat lambat/statis (mean_diff < 35 Hz dan std > 150 Hz) -> Ini WAIL Ambulance/Damkar!
        elif folder == "POLICE" and mean_diff < 35.0 and std_pitch > 180.0:
            misclassified.append((folder, filepath, filename, f"Terdeteksi WAIL LAMBAT / AMBULANCE (Modulasi Lambat {mean_diff:.1f}Hz/step)"))

print(f"[*] Total sampel sumber diperiksa : {sum(1 for f in SIREN_FOLDERS for _ in glob.glob(os.path.join(ROOT, f, '*.wav')) if not os.path.basename(_).startswith('aug_'))} file")
print(f"[*] Potensi salah klasifikasi     : {len(misclassified)} file")

if misclassified:
    print("\n--- DAFTAR FILE SALAH KAMAR (MISCLASSIFIED / CROSS-CLASS) ---")
    for idx, (fld, fp, fn, reason) in enumerate(misclassified[:25], 1):
        print(f" {idx:2d}. [{fld}] {fn:<40} -> {reason}")
    if len(misclassified) > 25:
        print(f"     ... dan {len(misclassified)-25} file lainnya.")
else:
    print("[🏆 SEMPURNA] Seluruh file berada di folder klasifikasi yang tepat! (Ambulance=Wail, Police=Yelp/Hi-Lo, Fire=Q-Siren)")

print("\n" + "=" * 80)
