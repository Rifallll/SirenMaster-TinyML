"""
audit_duplicates.py
===================
Skrip khusus untuk mendeteksi file duplikat di seluruh dataset SirenMaster
menggunakan algoritma hashing MD5 untuk menjamin akurasi 100%.
Mendeteksi:
1. Duplikat Internal (Intra-class): File kembar di dalam folder yang sama.
2. Duplikat Lintas Kelas (Cross-class): File yang sama persis berada di dua folder berbeda (SANGAT BERBAHAYA!).
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
            # Baca per 64KB untuk efisiensi RAM
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"  [!] Gagal membaca hash {os.path.basename(filepath)}: {e}")
        return None

print("=" * 70)
print("  AUDIT DETEKSI FILE DUPLIKAT DATASET SIRENMASTER (MD5 HASHING)")
print("=" * 70)

# Dictionary untuk menyimpan hash -> list path file
hash_map = defaultdict(list)
total_files = 0

print("[*] Menghitung MD5 Hash untuk semua audio di dataset...")
for cls in CLASSES:
    folder = os.path.join(ROOT, cls)
    files = glob.glob(os.path.join(folder, "*.wav")) + glob.glob(os.path.join(folder, "*.mp3"))
    print(f"    - Memindai kelas {cls}: {len(files)} file...")
    for f in files:
        h = calculate_md5(f)
        if h:
            hash_map[h].append((cls, f))
            total_files += 1

print(f"\n[*] Pemindaian selesai. Total file dipindai: {total_files}")
print("-" * 70)

# Kelompokkan jenis duplikat
intra_duplicates = []
cross_duplicates = defaultdict(list)
unique_hashes = 0
duplicate_files_count = 0

for h, file_list in hash_map.items():
    if len(file_list) > 1:
        # Ada duplikat!
        duplicate_files_count += (len(file_list) - 1)
        classes_involved = set(item[0] for item in file_list)
        
        if len(classes_involved) == 1:
            # Duplikat di dalam kelas yang sama
            cls = list(classes_involved)[0]
            intra_duplicates.append((cls, [item[1] for item in file_list]))
        else:
            # Duplikat lintas kelas (sangat berbahaya!)
            cross_duplicates[tuple(sorted(classes_involved))].append([item[1] for item in file_list])
    else:
        unique_hashes += 1

# --- CETAK LAPORAN ---
print(f"  HASIL AUDIT DATA:")
print(f"  - Total File Unik (Aman)      : {unique_hashes} file")
print(f"  - Total File Duplikat (Redund): {duplicate_files_count} file")
print("-" * 70)

if not intra_duplicates and not cross_duplicates:
    print("  [SUCCESS] Bersih 100%! Tidak ditemukan file duplikat sama sekali!")
    print("  Dataset Anda sangat sehat untuk training.")
    print("=" * 70)
    sys.exit(0)

# Tampilkan Duplikat Internal
if intra_duplicates:
    print(f"  [⚠️ WARNING] Ditemukan {len(intra_duplicates)} kelompok duplikat INTERNAL:")
    for cls, paths in intra_duplicates[:10]:
        print(f"    • Kelas [{cls}]:")
        for p in paths:
            print(f"      -> {os.path.basename(p)}")
        print()
    if len(intra_duplicates) > 10:
        print(f"    ... dan {len(intra_duplicates) - 10} kelompok lainnya.")
    print("-" * 70)

# Tampilkan Duplikat Lintas Kelas (Cross-class)
if cross_duplicates:
    print(f"  [🚨 BAHAYA!!!] Ditemukan file yang sama persis di kelas yang BERBEDA:")
    total_cross_groups = 0
    for cls_pair, groups in cross_duplicates.items():
        total_cross_groups += len(groups)
        print(f"    • Konflik antara kelas {cls_pair}:")
        for g in groups[:5]:
            for p in g:
                print(f"      -> {p.replace(ROOT, '')}")
            print("      " + "-"*30)
    print(f"    Total kelompok lintas kelas: {total_cross_groups}")
    print("  [!] Lintas kelas harus dihapus agar AI tidak bingung saat belajar!")
    print("-" * 70)

# --- REKOMENDASI DAN AKSI ---
print("\n[Rekomendasi Tindakan]:")
print("  - Untuk duplikat INTERNAL: Cukup pertahankan 1 file pertama, hapus sisanya agar training lebih cepat.")
print("  - Untuk duplikat LINTAS KELAS: Harus segera dihapus agar tidak merusak akurasi evaluasi model.")
print("=" * 70)
