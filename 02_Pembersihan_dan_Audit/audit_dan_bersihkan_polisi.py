"""
audit_dan_bersihkan_polisi.py
=============================
Audit Akustik Khusus Dataset POLICE (POLISI):
1. Mendeteksi file yang berisi suara manusia bicara, hening, atau bukan sirine Polisi.
2. Memisahkan & membersihkan file-file yang tidak sesuai.
3. Memastikan hanya file sirine Polisi murni (Yelp, Wail, Hi-Lo, Phaser) yang tersisa.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os
import glob
import shutil
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import tensorflow as tf

ROOT = r"C:\Users\ASUS\Videos\DATASET"
POLICE_DIR = os.path.join(ROOT, "POLICE")
QUARANTINE_DIR = os.path.join(ROOT, "KARANTINA_POLICE_NON_POLISI")
MODEL_PATH = os.path.join(ROOT, "siren_classifier_model.h5")

os.makedirs(QUARANTINE_DIR, exist_ok=True)

print("=" * 70)
print("🚓 AUDIT & PEMBERSIHAN DATASET POLISI (POLICE)")
print(f"📁 Folder Sumber     : {POLICE_DIR}")
print(f"📁 Folder Karantina  : {QUARANTINE_DIR}")
print("=" * 70)

# Load model AI untuk verifikasi silang
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[OK] Model AI dimuat untuk verifikasi silang.")
    except Exception as e:
        print(f"[!] Gagal memuat model: {e}")

SAMPLE_RATE = 8000
DURATION = 4.0
TARGET_LEN = int(SAMPLE_RATE * DURATION)
_hamming = np.hamming(256)
_mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=256, n_mels=40, fmin=0, fmax=4000)

def extract_features(y):
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)), mode='constant')
    else:
        y = y[:TARGET_LEN]
        
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y * min(1.0 / max_val, 10.0)
        
    y_smoothed = np.convolve(y, [1/3, 1/3, 1/3], mode='same')
    n_frames = (TARGET_LEN - 256) // 128 + 1
    
    log_mel = []
    for frame in range(n_frames):
        start = frame * 128
        frame_data = y_smoothed[start:start+256].copy()
        frame_data -= np.mean(frame_data)
        power_spec = np.abs(np.fft.rfft(frame_data * _hamming, n=256)) ** 2
        log_mel.append(np.log(np.dot(_mel_fb, power_spec) + 1e-9))
    return np.array(log_mel, dtype=np.float32)

# Load scaler constants
mean_c = None
std_c = None
with open(os.path.join(ROOT, "sirenmaster_main", "model.h"), "r") as f:
    import re
    c = f.read()
    mm = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\}', c)
    sm = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\}', c)
    if mm and sm:
        mean_c = np.array([float(x.strip().rstrip('f')) for x in mm.group(1).split(',')])
        std_c = np.array([float(x.strip().rstrip('f')) for x in sm.group(1).split(',')])

wav_files = sorted(glob.glob(os.path.join(POLICE_DIR, "*.wav")))
print(f"\n[*] Total file yang akan diperiksa: {len(wav_files)} file...\n")

quarantine_list = []
clean_list = []

for idx, filepath in enumerate(wav_files):
    fname = os.path.basename(filepath)
    
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    except Exception as e:
        quarantine_list.append((fname, f"Corrupt Audio: {str(e)}", filepath))
        continue
        
    # 1. Cek Durasi & Volume
    duration = len(y) / sr
    rms = np.sqrt(np.mean(y ** 2))
    
    if duration < 1.0:
        quarantine_list.append((fname, f"Durasi Terlalu Pendek ({duration:.2f}s)", filepath))
        continue
        
    if rms < 0.005:
        quarantine_list.append((fname, f"Terlalu Sunyi / Hening (RMS={rms:.4f})", filepath))
        continue
        
    # 2. Cek Karakteristik Akustik Sirine Polisi (600 - 3500 Hz)
    fft_mag = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    
    siren_band = (freqs >= 450) & (freqs <= 3500)
    low_band = (freqs < 450)
    
    siren_energy = np.sum(fft_mag[siren_band] ** 2)
    low_energy = np.sum(fft_mag[low_band] ** 2) + 1e-9
    total_energy = np.sum(fft_mag ** 2) + 1e-9
    
    siren_ratio = siren_energy / total_energy
    
    # Jika suara didominasi vokal/gumaman orang bicara (< 450 Hz)
    if siren_ratio < 0.30 and (low_energy / total_energy) > 0.60:
        quarantine_list.append((fname, f"Bukan Sirine Polisi / Suara Manusia-Mesin (Siren Ratio={siren_ratio*100:.1f}%)", filepath))
        continue
        
    # 3. Verifikasi Silang dengan Model AI
    if model is not None and mean_c is not None:
        feat = extract_features(y)
        feat_norm = (feat - mean_c) / std_c
        feat_norm = np.expand_dims(feat_norm, axis=(0, -1))
        preds = model.predict(feat_norm, verbose=0)[0]
        
        # Kelas: [0: AMBULANCE, 1: FIRETRUCK, 2: NORMAL, 3: POLICE]
        # Jika AI sangat yakin ini NORMAL (> 75%) dan Polisi < 12%
        if preds[2] > 0.75 and preds[3] < 0.12:
            quarantine_list.append((fname, f"AI Yakin NORMAL/Bising ({preds[2]*100:.1f}%)", filepath))
            continue
            
    clean_list.append((fname, filepath))

print("=" * 70)
print(f"📊 HASIL AUDIT DATASET POLISI (POLICE):")
print(f"   ✅ File Bersih & Valid  : {len(clean_list)} file")
print(f"   ⚠️ File Tidak Sesuai     : {len(quarantine_list)} file (dipindahkan ke karantina)")
print("=" * 70)

for fname, reason, fpath in quarantine_list:
    dest = os.path.join(QUARANTINE_DIR, fname)
    shutil.move(fpath, dest)
    print(f"  [KARANTINA] {fname:30} -> Alasan: {reason}")

print(f"\n🎯 Pemisahan selesai! {len(quarantine_list)} file tidak sesuai dipindahkan ke {QUARANTINE_DIR}.")
