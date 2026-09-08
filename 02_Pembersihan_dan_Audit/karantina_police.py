"""
Langkah 2: Pindahkan file POLICE bermasalah ke folder KARANTINA.
Tidak dihapus - hanya dipindahkan agar bisa dikembalikan jika perlu.
"""
import os, shutil

POLICE_DIR = r"C:\Users\ASUS\Videos\DATASET\POLICE"
KARANTINA_DIR = r"C:\Users\ASUS\Videos\DATASET\DATASET_ARCHIVE\POLICE_KARANTINA"
PROBLEMATIC_LIST = r"C:\Users\ASUS\Videos\DATASET\problematic_police_files.txt"

os.makedirs(KARANTINA_DIR, exist_ok=True)

# Baca daftar file bermasalah
with open(PROBLEMATIC_LIST, "r") as f:
    files = [line.strip() for line in f if line.strip()]

print(f"Total file akan dikarantina: {len(files)}")
print("=" * 60)

moved = 0
for fpath in files:
    fname = os.path.basename(fpath)
    dest = os.path.join(KARANTINA_DIR, fname)
    
    if not os.path.exists(fpath):
        print(f"  [SKIP] Tidak ada: {fname}")
        continue
    
    try:
        shutil.move(fpath, dest)
        print(f"  [PINDAH] {fname}")
        moved += 1
    except Exception as e:
        print(f"  [ERROR] {fname}: {e}")

# Cari juga augmentasi dari file bermasalah yang masih di POLICE
print()
print("[*] Mencari augmentasi dari file bermasalah...")
PROBLEM_ROOTS = [
    "police_0002", "police_0413", "police_0319", "police_0321",
    "police_0073", "police_0067", "police_0163", "police_0103",
    "sound_650", "sound_680"
]
aug_moved = 0
for fname in os.listdir(POLICE_DIR):
    if not fname.endswith('.wav'):
        continue
    for root in PROBLEM_ROOTS:
        if root in fname and ('aug_' in fname or '_noise_' in fname or '_loud' in fname or '_shift' in fname):
            src = os.path.join(POLICE_DIR, fname)
            dst = os.path.join(KARANTINA_DIR, fname)
            try:
                shutil.move(src, dst)
                print(f"  [PINDAH AUG] {fname}")
                aug_moved += 1
            except Exception as e:
                print(f"  [ERROR AUG] {fname}: {e}")
            break

print()
print("=" * 60)
print(f"SELESAI: {moved} file utama + {aug_moved} augmentasi dikarantina.")
print(f"Folder karantina: {KARANTINA_DIR}")
print()
print("Cache akan perlu dibangun ulang saat training berikutnya.")
