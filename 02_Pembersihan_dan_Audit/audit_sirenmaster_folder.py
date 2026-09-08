"""
audit_sirenmaster_folder.py
===========================
Audit lengkap folder sirenmaster_main: model.h, siren_model_data.h, 
referensi di .ino, sinkronisasi scaler, dan threshold detection.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, re, hashlib, numpy as np, datetime

ROOT = r"C:\Users\ASUS\Videos\DATASET\sirenmaster_main"
DATASET = r"C:\Users\ASUS\Videos\DATASET"

print("=" * 70)
print(" AUDIT LENGKAP: sirenmaster_main")
print("=" * 70)

# 1. Daftar file
print("\n[1] DAFTAR FILE:")
for f in sorted(os.listdir(ROOT)):
    path = os.path.join(ROOT, f)
    size = os.path.getsize(path)
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    print(f"    {f:<42} {size:>10} bytes  {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

# 2. Cek model.h
print("\n[2] CEK model.h:")
modelh = os.path.join(ROOT, "model.h")
content = open(modelh, "r", encoding="utf-8", errors="replace").read()

# Cek siren_model_data
match = re.search(r"siren_model_data\[\]\s*=\s*\{([^}]+)\}", content, re.DOTALL)
if match:
    hex_vals = re.findall(r"0x([0-9a-fA-F]{2})", match.group(1))
    model_bytes = bytes([int(h, 16) for h in hex_vals])
    md5_h = hashlib.md5(model_bytes).hexdigest()
    print(f"    siren_model_data: {len(model_bytes)} bytes, MD5={md5_h}")
else:
    print("    siren_model_data[] TIDAK DITEMUKAN di model.h!")

# Cek siren_model_data_len
len_match = re.search(r"siren_model_data_len\s*=\s*(\d+)", content)
if len_match:
    declared_len = int(len_match.group(1))
    actual_len = len(model_bytes) if match else 0
    ok = (declared_len == actual_len)
    status = "COCOK" if ok else "BERBEDA!"
    print(f"    siren_model_data_len declared: {declared_len}, actual: {actual_len} -> {status}")

# Cek MEL_MEAN/STD sync
scaler = np.load(os.path.join(DATASET, "siren_scaler.npz"))
g_mean = scaler["global_mean"]
g_std  = scaler["global_std"]
mean_m = re.findall(r"MEL_MEAN\[.*?PROGMEM\s*=\s*\{([^}]+)\}", content, re.DOTALL)
std_m  = re.findall(r"MEL_STD\[.*?PROGMEM\s*=\s*\{([^}]+)\}",  content, re.DOTALL)
if mean_m:
    vals = [v.strip().rstrip("f") for v in mean_m[0].strip().split(",") if v.strip()]
    h_mean = np.array([float(v) for v in vals if v])
    diff_mean = float(np.abs(h_mean - g_mean).max())
    status_mean = "SEMPURNA" if diff_mean < 1e-5 else "TIDAK COCOK!"
    print(f"    MEL_MEAN diff vs scaler: {diff_mean:.10f} -> {status_mean}")
if std_m:
    vals = [v.strip().rstrip("f") for v in std_m[0].strip().split(",") if v.strip()]
    h_std = np.array([float(v) for v in vals if v])
    diff_std = float(np.abs(h_std - g_std).max())
    status_std = "SEMPURNA" if diff_std < 1e-5 else "TIDAK COCOK!"
    print(f"    MEL_STD  diff vs scaler: {diff_std:.10f} -> {status_std}")

# 3. Cek siren_model_data.h
print("\n[3] CEK siren_model_data.h (file lama, jangan sampai terpakai!):")
smd_path = os.path.join(ROOT, "siren_model_data.h")
if os.path.exists(smd_path):
    smd_content = open(smd_path, "r", encoding="utf-8", errors="replace").read()
    arrays = re.findall(r"const\s+unsigned\s+char\s+(\w+)\[", smd_content)
    size = os.path.getsize(smd_path)
    print(f"    ADA! Array names: {arrays}, size: {size} bytes")
    # Cek apakah di-include di .ino
    ino_content = open(os.path.join(ROOT, "sirenmaster_main.ino"), "r", encoding="utf-8", errors="replace").read()
    if "siren_model_data.h" in ino_content:
        print("    BAHAYA: siren_model_data.h di-include di .ino!")
    else:
        print("    OK: TIDAK di-include di .ino (aman, tidak terpakai)")
else:
    print("    Tidak ada")

# 4. Cek model.tflite
print("\n[4] CEK model.tflite vs model.h:")
tflite_path = os.path.join(ROOT, "model.tflite")
tflite_train = os.path.join(DATASET, "04_Training_AI", "siren_model_quant.tflite")
if os.path.exists(tflite_path):
    tfl_data = open(tflite_path, "rb").read()
    md5_tfl = hashlib.md5(tfl_data).hexdigest()
    print(f"    model.tflite: {len(tfl_data)} bytes, MD5={md5_tfl}")
    if match:
        if md5_tfl == md5_h:
            print(f"    model.h == model.tflite: YA, COCOK SEMPURNA")
        else:
            print(f"    model.h == model.tflite: TIDAK COCOK! model.h={md5_h}, tflite={md5_tfl}")

# 5. Referensi di .ino
print("\n[5] REFERENSI KUNCI di sirenmaster_main.ino:")
ino_content = open(os.path.join(ROOT, "sirenmaster_main.ino"), "r", encoding="utf-8", errors="replace").read()
ino_lines = ino_content.split("\n")
keywords = ["#include", "GetModel", "siren_model_data", "MIC_GAIN", "CONFIDENCE_THR", 
            "rawScore >= 0.", "bp >= 0.", "weak_siren_streak >="]
for i, l in enumerate(ino_lines):
    for kw in keywords:
        if kw in l and l.strip() and not l.strip().startswith("//"):
            print(f"    L{i+1}: {l.strip()[:90]}")

# 6. Summary
print("\n[6] RINGKASAN AUDIT:")
tflite_correct_md5 = hashlib.md5(open(tflite_train, "rb").read()).hexdigest() if os.path.exists(tflite_train) else None
if tflite_correct_md5:
    print(f"    MD5 training tflite (benar): {tflite_correct_md5}")
    if match:
        match_ok = (md5_h == tflite_correct_md5)
        status = "COCOK -> model.h SUDAH BENAR" if match_ok else "TIDAK COCOK -> model.h MASIH LAMA!"
        print(f"    MD5 model.h siren_model_data: {md5_h}")
        print(f"    Match: {status}")
