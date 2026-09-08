"""
clean_duplicates.py
===================
Skrip otomatis untuk menghapus file duplikat internal (intra-class) di dataset.
Skrip ini menghitung MD5 hash dari setiap file dan hanya mempertahankan file unik pertama,
lalu menghapus file duplikat berikutnya agar dataset 100% bersih dan optimal.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import glob
import hashlib
from collections import defaultdict

ROOT = r"C:\Users\ASUS\Videos\DATASET"
CLASSES = ["AMBULANCE", "FIRETRUCK", "NORMAL", "POLICE"]

def calculate_md5(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        return None

print("=" * 70)
print("  PROSES CLEANING FILE DUPLIKAT DATASET SIRENMASTER")
print("=" * 70)

hash_map = defaultdict(list)
deleted_count = 0

for cls in CLASSES:
    folder = os.path.join(ROOT, cls)
    files = glob.glob(os.path.join(folder, "*.wav")) + glob.glob(os.path.join(folder, "*.mp3"))
    
    print(f"[*] Memindai {cls}...")
    for f in files:
        h = calculate_md5(f)
        if h:
            # Jika hash sudah ada di dalam kelompok ini, hapus duplikatnya!
            if h in hash_map:
                try:
                    os.remove(f)
                    deleted_count += 1
                except Exception as e:
                    print(f"  [!] Gagal menghapus {os.path.basename(f)}: {e}")
            else:
                hash_map[h].append(f)

print("-" * 70)
print(f"  [SUCCESS] Pembersihan selesai!")
print(f"  - Total File Duplikat Dihapus: {deleted_count} file")
print(f"  - Sisa Dataset Bersih/Unik    : {len(hash_map)} file")
print("=" * 70)
print("  Sekarang Anda bisa melakukan training ulang dengan dataset 100% unik!")
print("=" * 70)
