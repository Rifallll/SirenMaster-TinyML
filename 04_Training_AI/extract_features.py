# -*- coding: utf-8 -*-
"""
extract_features.py
-------------------
Ekstraksi MFCC‑13×63 dari semua file .wav dalam dataset
dan menyimpannya ke cache NPZ bernama
`siren_13x63_features_cache.npz`.

# Cara pakai (Windows command line):
#     python extract_features.py --src "C:\\Users\\ASUS\\Videos\\DATASET"
    python extract_features.py --src "C:\\Users\\ASUS\\Videos\\DATASET"
"""

import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm  # pip install tqdm if missing

# =============================================================================
# Parameter ekstraksi – harus sama dengan yang dipakai saat training / inferensi
# =============================================================================
SAMPLE_RATE = 8000     # 8 kHz, cocok untuk ESP32
N_FFT       = 256
HOP_LENGTH  = 128
N_MFCC      = 13
N_MELS      = 40           # tidak dipakai di MFCC, tapi dibutuhkan oleh librosa
DURATION    = 1.024        # detik (8192 sampel pada 8 kHz)
TARGET_LEN  = int(SAMPLE_RATE * DURATION)   # 8192 sampel

CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']


def pad_or_trim(y: np.ndarray) -> np.ndarray:
    """Pad dengan nol bila terlalu pendek atau potong bila terlalu panjang sehingga panjangnya persis TARGET_LEN."""
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)), mode='constant')
    else:
        y = y[:TARGET_LEN]
    return y


def extract_mfcc(y: np.ndarray) -> np.ndarray:
    """Mengembalikan MFCC berukuran (63, 13) – 63 frame, 13 koefisien."""
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=4000,
        window='hamming'
    )
    # librosa menghasilkan shape (13, 63); transpose supaya menjadi (63, 13)
    return mfcc.T.astype(np.float32)


def process_file(filepath: str) -> np.ndarray:
    """Membaca .wav, normalisasi panjang, dan mengembalikan fitur MFCC."""
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    y = pad_or_trim(y)
    return extract_mfcc(y)


def main(src_dir: str, out_path: str):
    """Iterate over all class folders, extract MFCCs and save to NPZ cache."""
    features = []   # list of np.ndarray (63,13)
    labels   = []   # integer label per CATEGORIES order

    for label_idx, cat in enumerate(CATEGORIES):
        class_dir = os.path.join(src_dir, cat)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Folder kelas tidak ditemukan: {class_dir}")
            continue
        wav_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.wav')]
        if not wav_files:
            print(f"[WARN] Tidak ada file wav di {class_dir}")
            continue
        print(f"[INFO] Memproses kelas {cat} ({len(wav_files)} file)")
        for fn in tqdm(wav_files, desc=f"  {cat}", leave=False):
            full_path = os.path.join(class_dir, fn)
            try:
                feat = process_file(full_path)   # (63,13)
                features.append(feat)
                labels.append(label_idx)
            except Exception as e:
                print(f"[ERROR] Gagal proses {full_path}: {e}")

    X = np.stack(features)            # (N,63,13)
    y = np.array(labels, dtype=np.int32)
    np.savez_compressed(out_path, X=X, y=y)
    print(f"\n[DONE] Cache disimpan di: {out_path}")
    print(f"  Samples   : {X.shape[0]}")
    print(f"  Shape tiap : {X.shape[1:]}  (63 × 13)")
    print(f"  Kelas     : {CATEGORIES}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstrak MFCC 13×63 untuk seluruh dataset")
    parser.add_argument("--src", type=str, required=True,
                        help="Folder dataset (harus berisi sub‑folder AMBULANCE, FIRETRUCK, NORMAL, POLICE)")
    parser.add_argument("--out", type=str, default="siren_13x63_features_cache.npz",
                        help="Nama file NPZ output (default: siren_13x63_features_cache.npz)")
    args = parser.parse_args()
    main(args.src, args.out)
