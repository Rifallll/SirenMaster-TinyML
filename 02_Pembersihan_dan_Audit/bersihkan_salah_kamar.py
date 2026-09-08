import os
import glob

DATASET_PATH = r"C:\Users\ASUS\Videos\DATASET"

# File kotor/salah kamar hasil audit cerdas (base name original)
CLEANUP_LIST = {
    "FIRETRUCK": [
        "fire_0003_seg105",
        "fire_0003_seg106",
        "fire_1000_seg01",
        "fire_1978_seg01",
        "fire_0998_seg01",
        "fire_0999_seg01",
        "fire_2076_seg01"
    ],
    "POLICE": [
        "police_0254_seg01"
    ],
    "AMBULANCE": [
        "ambulance_0210_seg01"
    ],
    "NORMAL": [
        "urban_0_13230-0-0-12",
        "urban_9_194733-9-0-14_4582"
    ]
}

print("="*60)
print("  MEMPROSES PEMBERSIHAN FILE SALAH KAMAR / KOTOR")
print("="*60)

total_deleted = 0

for cat, bases in CLEANUP_LIST.items():
    cat_dir = os.path.join(DATASET_PATH, cat)
    if not os.path.exists(cat_dir):
        continue
        
    for base in bases:
        # Cari file asli dan augmented yang mengandung base name tersebut
        # Contoh: fire_0003_seg105.wav atau aug_noise_1234_fire_0003_seg105.wav
        pattern = os.path.join(cat_dir, f"*{base}*.wav")
        matched_files = glob.glob(pattern)
        
        for fpath in matched_files:
            try:
                os.remove(fpath)
                print(f"  [-] Berhasil menghapus: [{cat}] {os.path.basename(fpath)}")
                total_deleted += 1
            except Exception as e:
                print(f"  [!] Gagal menghapus {os.path.basename(fpath)}: {e}")

print("="*60)
print(f"  SELESAI! Total file kotor & augmented dihapus: {total_deleted}")
print("="*60)
