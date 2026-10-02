"""
test_noise.py
=============
Memutar setiap file suara bising dari NOISE_TEST ke speaker, 
lalu pantau Serial Monitor untuk cek apakah alat false alarm atau tidak.

Kegunaan:
- Cek alat ESP32 tetap [SAFE] saat suara gergaji/ngelas dimainkan keras-keras
- Bandingkan dengan suara sirene asli yang WAJIB terdeteksi
"""
import sys, os, glob, time, winsound, subprocess

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = r"C:\Users\ASUS\Videos\DATASET"
NOISE_DIR = os.path.join(ROOT, "NOISE_TEST")

noisy_files = sorted(glob.glob(os.path.join(NOISE_DIR, "*.wav")))
print(f"Total file suara bising: {len(noisy_files)}")
print()

# Kelompokkan per kategori
categories = {}
for f in noisy_files:
    base = os.path.basename(f)
    for cat in ['gergaji_mesin2', 'gergaji_mesin', 'ngelas', 'mesin_tekuk']:
        if base.startswith(cat):
            categories.setdefault(cat, []).append(f)
            break

menu = """
============================================================
  UJI SUARA BISING vs ALAT SirenMaster ESP32
============================================================
Pilih suara yang ingin diputar:

  [1] Gergaji Mesin (Chainsaw) - 47 segmen
  [2] Ngelas (Welding) - 20 segmen  
  [3] Mesin Tekuk (Bending Machine) - 4 segmen
  [4] Putar SEMUA satu per satu (slow, satu segmen tiap kategori)
  [0] Keluar

Pantau layar LCD atau Serial Monitor Arduino saat diputar!
Alat HARUS tetap hijau [SAFE] untuk semua suara bising ini.
============================================================
"""

cat_map = {
    '1': ('gergaji_mesin', 'GERGAJI MESIN'),
    '2': ('ngelas', 'NGELAS / WELDING'),
    '3': ('mesin_tekuk', 'MESIN TEKUK'),
}

while True:
    print(menu)
    pilihan = input("Masukkan pilihan (0-4): ").strip()

    if pilihan == '0':
        print("Selesai.")
        break

    if pilihan == '4':
        # Putar 1 segmen dari setiap kategori
        for cat_key, label in cat_map.values():
            files = categories.get(cat_key, [])
            if not files:
                continue
            f = files[0]
            print(f"\n[MEMUTAR] {label}: {os.path.basename(f)}")
            print("  --> Perhatikan layar LCD alat! Harus tetap HIJAU [SAFE]")
            winsound.PlaySound(f, winsound.SND_FILENAME)
            time.sleep(1)
            print("  Selesai.")
        print("\nSemua kategori sudah dimainkan!")
        continue

    if pilihan not in cat_map:
        print("Pilihan tidak valid!")
        continue

    cat_key, label = cat_map[pilihan]
    files = categories.get(cat_key, [])
    if not files:
        print(f"Tidak ada file untuk {label}")
        continue

    print(f"\n[{label}] - {len(files)} segmen tersedia")
    print("Memilih 3 segmen terpanjang untuk diputar...")
    print()

    # Putar 3 segmen pertama
    for i, f in enumerate(files[:3]):
        print(f"  [{i+1}/3] {os.path.basename(f)}")
        print("  --> Pantau Serial Monitor / LCD alat! Harus [SAFE]!")
        winsound.PlaySound(f, winsound.SND_FILENAME)
        time.sleep(0.5)
        inp = input("  Alat masih [SAFE]? (y=ya/n=ada false alarm): ").strip().lower()
        if inp == 'n':
            print("  !! FALSE ALARM terdeteksi - perlu dicatat !!")
        else:
            print("  Bagus! Alat tetap [SAFE].")
        print()
