"""
regenerate_model_h.py
=====================
Regenerate model.h LENGKAP dari model TFLite yang benar:
- siren_model_data[] = bytes dari model TFLite yang sudah terbukti 100% akurat
- HAMMING_WINDOW[] = dari librosa
- MEL_FILTERBANK[][] = dari librosa
- MEL_MEAN[] dan MEL_STD[] = dari siren_scaler.npz (SYNC SEMPURNA)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, re, numpy as np, librosa

ROOT = r"C:\Users\ASUS\Videos\DATASET"
TFLITE = os.path.join(ROOT, "04_Training_AI", "siren_model_quant.tflite")
SCALER = os.path.join(ROOT, "siren_scaler.npz")
OUTPUT = os.path.join(ROOT, "sirenmaster_main", "model.h")

print("=" * 60)
print(" REGENERATE model.h LENGKAP")
print("=" * 60)

# 1. Baca model TFLite bytes
model_bytes = open(TFLITE, 'rb').read()
print(f"[1] Model TFLite: {len(model_bytes)} bytes dari {os.path.basename(TFLITE)}")

# 2. Baca scaler
scaler = np.load(SCALER)
g_mean = scaler['global_mean']  # shape (40,)
g_std  = scaler['global_std']   # shape (40,)
print(f"[2] Scaler: mean={g_mean[0]:.6f}, std={g_std[0]:.6f}")

# 3. Hitung Hamming window
N_FFT = 256
hamming = np.hamming(N_FFT)
print(f"[3] Hamming window: {len(hamming)} poin")

# 4. Hitung Mel filterbank
N_MELS = 40
mel_fb = librosa.filters.mel(sr=8000, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000)
print(f"[4] Mel filterbank: {mel_fb.shape}")

# 5. Generate C header
def floats_to_c(arr, name, dims):
    lines = [f"const float {name}{dims} PROGMEM = {{"]
    flat = arr.flatten()
    row = []
    for i, v in enumerate(flat):
        row.append(f"{v:.8f}f")
        if len(row) == 8 or i == len(flat) - 1:
            lines.append("  " + ", ".join(row) + ("," if i < len(flat)-1 else ""))
            row = []
    lines.append("};")
    return "\n".join(lines)

def bytes_to_c(data, name):
    lines = [f"const unsigned int {name}_len = {len(data)};",
             "",
             f"#ifdef __AVR__",
             f"  const unsigned char PROGMEM {name}[] = {{",
             f"#else",
             f"  __attribute__((aligned(4)))",
             f"  const unsigned char {name}[] = {{",
             f"#endif"]
    row = []
    for i, b in enumerate(data):
        row.append(f"0x{b:02x}")
        if len(row) == 12 or i == len(data)-1:
            lines.append("  " + ", ".join(row) + ("," if i < len(data)-1 else ""))
            row = []
    lines.append("};")
    return "\n".join(lines)

header = """/*
 * model.h - Siren Classifier (v2 Police Fix) - REGENERATED
 * Auto-generated: siren_model_data dari siren_model_quant.tflite
 * MEL_MEAN/STD   dari siren_scaler.npz (SYNC SEMPURNA)
 */

#ifndef MODEL_H
#define MODEL_H

#include <pgmspace.h>

#define SAMPLE_RATE_HZ   8000
#define N_FFT_SIZE       256
#define N_HOP_LENGTH     128
#define N_MFCC_COEFF     13
#define N_MEL_FILTERS    40
#define N_TIME_FRAMES    249
#define N_FFT_BINS       129
#define AUDIO_SAMPLES    32000
#define NUM_CLASSES      4

#ifndef MODEL_ALIGN
#define MODEL_ALIGN __attribute__((aligned(4)))
#endif

"""

hamming_c   = floats_to_c(hamming, "HAMMING_WINDOW", "[256]")
mel_c       = floats_to_c(mel_fb, "MEL_FILTERBANK", "[40][129]")
mean_c      = floats_to_c(g_mean, "MEL_MEAN", "[40]")
std_c       = floats_to_c(g_std,  "MEL_STD",  "[40]")
model_c     = bytes_to_c(model_bytes, "siren_model_data")

full_content = (
    header
    + hamming_c + "\n\n"
    + mel_c + "\n\n"
    + mean_c + "\n\n"
    + std_c + "\n\n"
    + model_c + "\n\n"
    + "#endif // MODEL_H\n"
)

# Backup dulu
import shutil
backup = OUTPUT + ".bak2"
shutil.copy2(OUTPUT, backup)
print(f"[5] Backup: {backup}")

# Tulis model.h baru
with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write(full_content)

print(f"[6] model.h ditulis: {len(full_content)} chars, {OUTPUT}")
print()
print("=" * 60)
print(" VERIFIKASI ULANG:")
print("=" * 60)

# Verifikasi: baca balik dan cek
import hashlib, re as re2
content2 = open(OUTPUT, 'r', encoding='utf-8').read()
match = re2.search(r'siren_model_data\[\]\s*=\s*\{([^}]+)\}', content2, re2.DOTALL)
if match:
    hex_vals = re2.findall(r'0x([0-9a-fA-F]{2})', match.group(1))
    read_back = bytes([int(h,16) for h in hex_vals])
    md5_orig = hashlib.md5(model_bytes).hexdigest()
    md5_read = hashlib.md5(read_back).hexdigest()
    ok = (md5_orig == md5_read)
    print(f"MD5 original tflite : {md5_orig}")
    print(f"MD5 dibaca dari h    : {md5_read}")
    print(f"MATCH: {'YA COCOK SEMPURNA ✅' if ok else 'TIDAK COCOK ❌ - Ada bug!'}")
else:
    print("ERROR: siren_model_data tidak ditemukan di model.h baru!")

mean_match = re2.findall(r'MEL_MEAN\[.*?PROGMEM\s*=\s*\{([^}]+)\}', content2, re2.DOTALL)
if mean_match:
    vals = [v.strip().rstrip('f') for v in mean_match[0].strip().split(',') if v.strip()]
    modelh_mean = np.array([float(v) for v in vals if v])
    diff = np.abs(modelh_mean - g_mean).max()
    print(f"MEL_MEAN diff vs scaler: {diff:.10f} -> {'SEMPURNA ✅' if diff < 1e-6 else 'ADA PERBEDAAN ❌'}")

print()
print(">>> SEKARANG UPLOAD ULANG ke Arduino IDE! <<<")
