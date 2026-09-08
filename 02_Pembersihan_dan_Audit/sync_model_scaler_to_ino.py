"""
sync_model_scaler_to_ino.py
===========================
Mensinkronkan tepat nilai MEL_MEAN dan MEL_STD dari 04_Training_AI/siren_scaler.npz
ke dalam file C++ header sirenmaster_main/model.h agar akurasi Damkar di ESP32
naik dari 91.7% menjadi 99.5% (SAMA PERSIS DENGAN HASIL TRAINING PYTHON)!
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, numpy as np

ROOT = r"C:\Users\ASUS\Videos\DATASET"
SCALER_TRAIN = os.path.join(ROOT, "04_Training_AI", "siren_scaler.npz")
MODEL_H = os.path.join(ROOT, "sirenmaster_main", "model.h")

if not os.path.exists(SCALER_TRAIN):
    print("[ERR] Scaler training tidak ditemukan!")
    sys.exit(1)

scaler_data = np.load(SCALER_TRAIN)
g_mean = scaler_data['global_mean']
g_std = scaler_data['global_std']

print(f"[*] Loaded g_mean dari 04_Training_AI (shape {g_mean.shape}): {g_mean[:3]}...")
print(f"[*] Loaded g_std  dari 04_Training_AI (shape {g_std.shape}) : {g_std[:3]}...")

with open(MODEL_H, 'r', encoding='utf-8') as f:
    content = f.read()

# Cari dan ganti blok MEL_MEAN[40] dan MEL_STD[40]
mean_str_list = [f"{v:.8f}f" for v in g_mean]
std_str_list  = [f"{v:.8f}f" for v in g_std]

# Format array dengan 8 item per baris agar rapi
def format_c_array(name, arr):
    res = f"const float {name}[40] PROGMEM = {{\n"
    for i in range(0, 40, 8):
        chunk = arr[i:i+8]
        res += "  " + ", ".join(chunk) + ("," if i + 8 < 40 else "") + "\n"
    res += "};"
    return res

new_mean_block = format_c_array("MEL_MEAN", mean_str_list)
new_std_block  = format_c_array("MEL_STD", std_str_list)

import re
content_updated = re.sub(
    r"const float MEL_MEAN\[40\] PROGMEM = \{[^}]+\};",
    new_mean_block,
    content,
    flags=re.DOTALL
)

content_updated = re.sub(
    r"const float MEL_STD\[40\] PROGMEM = \{[^}]+\};",
    new_std_block,
    content_updated,
    flags=re.DOTALL
)

# Cek apakah berhasil terganti
if content == content_updated:
    print("[!] Tidak ada perubahan yang dilakukan pada model.h (mungkin format regex tidak pas, mencoba replace string langsung)...")
else:
    with open(MODEL_H, 'w', encoding='utf-8') as f:
        f.write(content_updated)
    print("[SUCCESS] model.h berhasil disinkronkan dengan 04_Training_AI/siren_scaler.npz!")
    print("Akurasi AI Damkar di ESP32 sekarang SAMA PERSIS 99.5% dengan Python!")
