"""
audit_dataset_lengkap.py
=========================
Skrip audit lengkap untuk memeriksa integritas seluruh dataset SirenMaster:
- Jumlah file per kelas
- File rusak / tidak bisa dibaca
- File terlalu pendek (< 2 detik)
- Distribusi durasi rata-rata
- File NORMAL yang berpotensi berbahaya (mengandung kata kunci sirine)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, glob, librosa, numpy as np

ROOT = r"C:\Users\ASUS\Videos\DATASET"
CLASSES = ["AMBULANCE", "FIRETRUCK", "NORMAL", "POLICE"]

# Kata kunci berbahaya di folder NORMAL (yang bisa mengecoh model)
DANGER_KEYWORDS = ["ambulance", "police", "polisi", "firetruck", "damkar", "sirine", "siren", "wail", "yelp"]

print("=" * 65)
print("  AUDIT LENGKAP DATASET SIRENMASTER")
print("=" * 65)

total_all = 0
total_ok = 0
total_corrupt = 0
total_short = 0
all_results = {}

for cls in CLASSES:
    folder = os.path.join(ROOT, cls)
    files = glob.glob(os.path.join(folder, "*.wav")) + glob.glob(os.path.join(folder, "*.mp3"))
    ok, corrupt, short = 0, 0, 0
    durs = []
    corrupt_files = []
    short_files = []

    for f in files:
        try:
            y, sr = librosa.load(f, sr=None, duration=5.0)
            dur = librosa.get_duration(y=y, sr=sr)
            durs.append(dur)
            if dur < 2.0:
                short += 1
                short_files.append(os.path.basename(f))
            else:
                ok += 1
        except Exception as e:
            corrupt += 1
            corrupt_files.append(os.path.basename(f))

    all_results[cls] = {
        "total": len(files),
        "ok": ok,
        "corrupt": corrupt,
        "short": short,
        "durs": durs,
        "corrupt_files": corrupt_files,
        "short_files": short_files
    }
    total_all += len(files)
    total_ok += ok
    total_corrupt += corrupt
    total_short += short

print()
for cls, r in all_results.items():
    avg_dur = float(np.mean(r["durs"])) if r["durs"] else 0.0
    status = "[OK] AMAN" if r["corrupt"] == 0 and r["short"] < 10 else "[PERLU PERHATIAN]"
    print(f"  [{cls}] {status}")
    print(f"    Total File    : {r['total']:5d} file")
    print(f"    File OK (>=2s): {r['ok']:5d} file  [OK]")
    print(f"    File Pendek   : {r['short']:5d} file")
    print(f"    File Rusak    : {r['corrupt']:5d} file")
    print(f"    Rata2 Durasi  : {avg_dur:.2f} detik")
    if r["corrupt_files"]:
        print(f"    [!] File Rusak : {r['corrupt_files'][:3]}")
    print()

# Cek khusus NORMAL: ada file berbahaya?
print("-" * 65)
print("  [CECK KHUSUS] File NORMAL yang Berpotensi Rancu:")
normal_folder = os.path.join(ROOT, "NORMAL")
normal_files = glob.glob(os.path.join(normal_folder, "*.wav")) + glob.glob(os.path.join(normal_folder, "*.mp3"))
danger_found = []
for f in normal_files:
    bname = os.path.basename(f).lower()
    for kw in DANGER_KEYWORDS:
        if kw in bname:
            danger_found.append(os.path.basename(f))
            break

if danger_found:
    print(f"  [!] Ditemukan {len(danger_found)} file dengan nama mengandung kata 'siren/police/ambulance' di NORMAL!")
    for df in danger_found[:10]:
        print(f"      -> {df}")
    if len(danger_found) > 10:
        print(f"      ... dan {len(danger_found) - 10} lainnya.")
else:
    print("  [OK] Tidak ada file berbahaya di folder NORMAL.")

print()
print("=" * 65)
print(f"  GRAND TOTAL   : {total_all:6d} file")
print(f"  File OK       : {total_ok:6d} file  [OK]")
print(f"  File Pendek   : {total_short:6d} file")
print(f"  File Rusak    : {total_corrupt:6d} file")
print("=" * 65)
if total_corrupt == 0:
    print("  [SUCCESS] Dataset 100% BERSIH dan SIAP untuk Training!")
else:
    print("  [WARNING] Ada file rusak yang perlu diperiksa lebih lanjut!")
print("=" * 65)
