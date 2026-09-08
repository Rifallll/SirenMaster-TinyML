"""
audit_sirene_anomali.py
=======================
Audit Otomatis Berbasis Spektrogram & Energi Akustik untuk Mendeteksi
File Aneh / Bukan Sirene / Intro Diam / Corrupt di Folder AMBULANCE, FIRETRUCK, dan POLICE.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, glob, numpy as np
import scipy.io.wavfile as wavfile

ROOT = r"C:\Users\ASUS\Videos\DATASET"
SIREN_FOLDERS = ["AMBULANCE", "FIRETRUCK", "POLICE"]

print("=" * 75)
print(" 🔍 AUDIT SPEKTROGRAM & ENERGI AKUSTIK KELAS SIRENE")
print("    Mendeteksi file diam (silence), intro/speech, dan anomali spektrogram...")
print("=" * 75)

total_scanned = 0
anomalies_found = {
    "silent": [],
    "corrupt_short": [],
    "extreme_clipping": [],
    "non_siren_spectral": []
}

for folder in SIREN_FOLDERS:
    folder_path = os.path.join(ROOT, folder)
    wav_files = sorted(glob.glob(os.path.join(folder_path, "*.wav")))
    print(f"\n[*] Memeriksa {len(wav_files)} file di folder {folder}...")
    
    for filepath in wav_files:
        total_scanned += 1
        filename = os.path.basename(filepath)
        
        try:
            sr, data = wavfile.read(filepath)
        except Exception as e:
            anomalies_found["corrupt_short"].append((folder, filename, f"Corrupt Header: {str(e)}"))
            continue
            
        # Konversi ke float mono
        if len(data.shape) > 1:
            data = data[:, 0]
        data_float = data.astype(np.float32) / 32768.0
        
        # 1. Cek Durasi terlalu pendek (< 0.5 detik)
        duration = len(data_float) / sr
        if duration < 0.5:
            anomalies_found["corrupt_short"].append((folder, filename, f"Durasi pendek ({duration:.2f}s)"))
            continue
            
        # 2. Cek Keheningan (Silent / Intro Diam)
        rms = np.sqrt(np.mean(data_float ** 2))
        if rms < 0.005: # Sangat hening (< 0.5% amplitudo)
            anomalies_found["silent"].append((folder, filename, f"Hening/Intro Diam (RMS={rms:.4f})"))
            continue
            
        # 3. Cek Extreme Clipping (Distorsi rusak / square wave > 40% sample mentok)
        clip_ratio = np.sum(np.abs(data_float) >= 0.99) / len(data_float)
        if clip_ratio > 0.40:
            anomalies_found["extreme_clipping"].append((folder, filename, f"Clipping Parah ({clip_ratio*100:.1f}%)"))
            continue
            
        # 4. Cek Spektrogram & Tonalitas (Anomali Bukan Sirene)
        # Sirene memiliki energi terkonsentrasi di band 300 Hz - 3000 Hz.
        # Kita lakukan FFT sederhana untuk melihat distribusi energi spektrogram.
        fft_spec = np.abs(np.fft.rfft(data_float)) ** 2
        freqs = np.fft.rfftfreq(len(data_float), 1.0 / sr)
        
        total_energy = np.sum(fft_spec)
        if total_energy == 0:
            anomalies_found["silent"].append((folder, filename, "Energi FFT Nol"))
            continue
            
        # Hitung persentase energi di bawah 100 Hz (gemuruh angin / AC / mic bump)
        low_energy_ratio = np.sum(fft_spec[freqs < 100]) / total_energy
        # Hitung persentase energi di band sirene (300 Hz - 3000 Hz)
        siren_band_ratio = np.sum(fft_spec[(freqs >= 300) & (freqs <= 3000)]) / total_energy
        
        # Jika energi sirene sangat rendah (< 12%) atau dominasi angin/gemuruh bawah 100Hz (> 85%),
        # ini kemungkinan besar bukan suara sirene (melainkan intro bicara/angin/hening berderau)
        if siren_band_ratio < 0.12 or low_energy_ratio > 0.85:
            anomalies_found["non_siren_spectral"].append(
                (folder, filename, f"Energi Sirene Rendah ({siren_band_ratio*100:.1f}%, Low-Hz={low_energy_ratio*100:.1f}%)")
            )

# --- LAPORAN HASIL AUDIT ---
print("\n" + "=" * 75)
print(" 📊 LAPORAN HASIL AUDIT ANOMALI DATASET SIRENE")
print("=" * 75)
print(f"[*] Total file yang diperiksa : {total_scanned} file")

total_anomalies = sum(len(v) for v in anomalies_found.values())
print(f"[*] Total file mencurigakan   : {total_anomalies} file")

if total_anomalies == 0:
    print("\n[🏆 LUAR BIASA] Tidak ditemukan file aneh, diam, atau corrupt! Semua file 100% sehat & sesuai sirene.")
else:
    for cat, items in anomalies_found.items():
        if items:
            print(f"\n--- {cat.upper().replace('_', ' ')} ({len(items)} file) ---")
            for i, (fld, fn, reason) in enumerate(items[:15], 1):
                print(f"  {i}. [{fld}] {fn} -> {reason}")
            if len(items) > 15:
                print(f"     ... dan {len(items)-15} file lainnya.")

print("=" * 75)
