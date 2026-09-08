"""
Hapus file FIRETRUCK yang masih dikira AMBULANCE >= 50% keyakinan
Berdasarkan hasil scan_fire_vs_ambu.py — data asli
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = r"C:\Users\ASUS\Videos\DATASET\FIRETRUCK"

# File-file bermasalah dari hasil scan — urutkan dari paling yakin salah
HAPUS = [
    # >= 80% dikira AMBULANCE — JELAS salah masuk folder
    "fire_2069_seg01.wav",
    "guru_firetruck_1780558695656.wav",   # rekaman guru yang salah!
    "fire_1687_seg01.wav",
    "yt_indo_0_part_24.wav",
    "fire_1018_seg01.wav",
    "fire_1019_seg01.wav",
    "guru_firetruck_1780558546437.wav",   # rekaman guru yang salah!
    "yt_indo_0_part_23.wav",
    # >= 60% dikira AMBULANCE
    "fire_0004_seg06.wav",
    "fire_1699_seg01.wav",
    "fire_1974_seg01.wav",
    "fire_0890_seg01.wav",
    "fire_0220_seg01.wav",
    "sound_206.wav",
    "fire_1671_seg01.wav",
    "fire_0254_original.wav",
    "sound_394_1.wav",
    "fire_1692_seg01.wav",
    "fire_0004_seg08.wav",
    "fire_0254_noise_light.wav",
    "yt_indo_0_part_25.wav",
    "fire_0054_shift.wav",
    "fire_0417_original.wav",
    "fire_0275_original.wav",
    "fire_0275_noise_light.wav",
    # >= 50% dikira AMBULANCE
    "sound_209_1.wav",
    "fire_1893_seg01.wav",
    "sound_347.wav",
    "fire_1103_seg01.wav",
    "fire_0218_seg01.wav",
    "fire_0054_noise_light.wav",
    "fire_0134_seg01.wav",
    "fire_1122_seg01.wav",
    "fire_0280_noise_light.wav",
    "fire_0311_seg01.wav",
    "fire_1104_seg01.wav",
    "fire_0004_seg03.wav",
    "fire_0004_seg09.wav",
    "yt_indo_0_part_8.wav",
    "yt_indo_0_part_11.wav",
    "fire_1685_seg01.wav",
]

print("=" * 60)
print("  HAPUS FIRETRUCK YANG MASIH DIKIRA AMBULANCE")
print("=" * 60)

ok = 0
skip = 0
for fname in HAPUS:
    fp = os.path.join(BASE, fname)
    if os.path.exists(fp):
        os.remove(fp)
        print(f"  [OK] {fname}")
        ok += 1
    else:
        print(f"  [SKIP] {fname} (sudah tidak ada)")
        skip += 1

# Cek sisa FIRETRUCK
total = len([f for f in os.listdir(BASE) if f.lower().endswith(('.wav','.m4a'))])

print(f"\n{'='*60}")
print(f"  Dihapus  : {ok} file")
print(f"  Dilewati : {skip} file")
print(f"  Sisa FIRETRUCK: {total} file")
print(f"{'='*60}")
print(f"\n✅ Selesai! Jalankan: python train_lokal.py")
