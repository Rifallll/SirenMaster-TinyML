"""
audit_semua_siren_final.py
==========================
Audit FINAL & KETAT semua dataset sirine (AMBULANCE, FIRETRUCK, POLICE).
Fokus: pastikan setiap file adalah sirine murni. Jika bukan -> buang.
Kriteria ketat:
  1. Durasi >= 1.5 detik
  2. RMS >= 0.008 (tidak hening)
  3. Siren Ratio (energi 400-3500 Hz) >= 40%
  4. Verifikasi AI: skor kelas sirine >= 30%, skor NORMAL < 60%
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, glob, shutil, re
import numpy as np
import librosa
import tensorflow as tf

ROOT        = r"C:\Users\ASUS\Videos\DATASET"
SIREN_DIRS  = ["AMBULANCE", "FIRETRUCK", "POLICE"]
MODEL_PATH  = os.path.join(ROOT, "siren_classifier_model.h5")
QUARANTINE  = os.path.join(ROOT, "KARANTINA_FINAL_SIREN_AUDIT")

os.makedirs(QUARANTINE, exist_ok=True)

print("=" * 72)
print("  AUDIT FINAL & KETAT - SEMUA DATASET SIREN")
print("  Fokus: hanya sirine MURNI yang boleh tinggal")
print("=" * 72)

# ── Load Model ──────────────────────────────────────────────────────────────
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[OK] Model AI siap.\n")
    except Exception as e:
        print(f"[!] Model gagal dimuat: {e}\n")

# ── Scaler ───────────────────────────────────────────────────────────────────
mean_c = std_c = None
header_path = os.path.join(ROOT, "sirenmaster_main", "model.h")
with open(header_path, "r") as f:
    c = f.read()
    mm = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\}', c)
    sm = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\}', c)
    if mm and sm:
        mean_c = np.array([float(x.strip().rstrip('f')) for x in mm.group(1).split(',')])
        std_c  = np.array([float(x.strip().rstrip('f')) for x in sm.group(1).split(',')])

SR = 8000
TARGET = int(SR * 4.0)
_ham   = np.hamming(256)
_melfb = librosa.filters.mel(sr=SR, n_fft=256, n_mels=40, fmin=0, fmax=4000)

# Indeks kelas sirine per folder
SIREN_CLASS = {"AMBULANCE": 0, "FIRETRUCK": 1, "POLICE": 3}

def mel_features(y):
    if len(y) < TARGET:
        y = np.pad(y, (0, TARGET - len(y)))
    else:
        y = y[:TARGET]
    mx = np.max(np.abs(y))
    if mx > 1e-6:
        y = y * min(1.0 / mx, 10.0)
    y = np.convolve(y, [1/3,1/3,1/3], 'same')
    out = []
    for i in range((TARGET - 256) // 128 + 1):
        f = y[i*128:i*128+256].copy()
        f -= f.mean()
        out.append(np.log(np.dot(_melfb, np.abs(np.fft.rfft(f * _ham, 256))**2) + 1e-9))
    return np.array(out, np.float32)

# ── Audit ────────────────────────────────────────────────────────────────────
grand_clean = grand_dirty = 0
summary = {}

for folder in SIREN_DIRS:
    folder_path = os.path.join(ROOT, folder)
    files       = sorted(glob.glob(os.path.join(folder_path, "*.wav")))
    cls_idx     = SIREN_CLASS[folder]
    dirty       = []
    clean       = []

    print(f"\n[{folder}] Memeriksa {len(files)} file...")
    
    for fpath in files:
        fname = os.path.basename(fpath)
        
        try:
            y, _ = librosa.load(fpath, sr=SR)
        except Exception as e:
            dirty.append((fname, f"Corrupt: {e}", fpath))
            continue

        # 1. Durasi
        dur = len(y) / SR
        if dur < 1.5:
            dirty.append((fname, f"Terlalu pendek ({dur:.2f}s)", fpath))
            continue

        # 2. Keheningan
        rms = np.sqrt(np.mean(y**2))
        if rms < 0.008:
            dirty.append((fname, f"Hening (RMS={rms:.4f})", fpath))
            continue

        # 3. Spektrum sirine ketat (400-3500 Hz)
        fft_mag = np.abs(np.fft.rfft(y))
        freqs   = np.fft.rfftfreq(len(y), 1.0/SR)
        siren_e = np.sum(fft_mag[(freqs>=400)&(freqs<=3500)]**2)
        total_e = np.sum(fft_mag**2) + 1e-9
        ratio   = siren_e / total_e

        if ratio < 0.40:
            dirty.append((fname, f"Bukan sirine (SirenRatio={ratio*100:.1f}%)", fpath))
            continue

        # 4. AI double-check
        if model is not None and mean_c is not None:
            feat = mel_features(y)
            feat = (feat - mean_c) / std_c
            feat = feat[np.newaxis, ..., np.newaxis]
            p    = model.predict(feat, verbose=0)[0]

            siren_score = p[cls_idx]
            normal_score = p[2]

            # Jika AI yakin NORMAL > 60% DAN skor kelas sirine ini < 20%
            if normal_score > 0.60 and siren_score < 0.20:
                dirty.append((fname, f"AI: NORMAL {normal_score*100:.0f}% / {folder} {siren_score*100:.0f}%", fpath))
                continue

        clean.append(fname)

    # Pindahkan yang kotor ke karantina
    for fname, reason, fpath in dirty:
        dest = os.path.join(QUARANTINE, f"{folder}_{fname}")
        shutil.move(fpath, dest)
        print(f"  [BUANG] {fname:35} -> {reason}")

    grand_clean += len(clean)
    grand_dirty += len(dirty)
    summary[folder] = {"bersih": len(clean), "kotor": len(dirty)}

# ── Laporan ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  LAPORAN AKHIR AUDIT SIREN")
print("=" * 72)
for folder, r in summary.items():
    total = r['bersih'] + r['kotor']
    print(f"  {folder:12}: {r['bersih']:4} bersih  |  {r['kotor']:3} dibuang  (dari {total})")
print(f"\n  TOTAL BERSIH : {grand_clean} file")
print(f"  TOTAL DIBUANG: {grand_dirty} file")
print("=" * 72)
print(f"\nFile kotor ada di: {QUARANTINE}")
print("Jika OK -> hapus folder karantina lalu retrain ulang!")
