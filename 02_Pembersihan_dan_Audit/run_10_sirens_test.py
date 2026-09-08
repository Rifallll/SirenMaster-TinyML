"""
run_10_sirens_test.py
=====================
Skrip pengujian interaktif untuk memutar 10 jenis sirene jalanan nyata (WAV)
melalui speaker PC selama 8 detik per suara (diputar ulang 2 kali).
Berguna untuk simulasi langsung di hadapan Dosen Penguji / Ujian Skripsi!
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, time, glob, winsound

ROOT = r"C:\Users\ASUS\Videos\DATASET"

# 1. Daftarkan 10 file audio terpilih dari masing-masing folder
# Memilih sampel-sampel yang memiliki karakteristik frekuensi paling kuat & jelas
TEST_SAMPLES = [
    # [Target Kelas, Nama Folder, Sub-string Nama File]
    ("AMBULANCE", "AMBULANCE", "amb_new"),
    ("DAMKAR (FIRETRUCK)", "FIRETRUCK", "fire_0"),
    ("POLISI", "POLICE", "pol_"),
    ("AMBULANCE", "AMBULANCE", "amb_new"),
    ("DAMKAR (FIRETRUCK)", "FIRETRUCK", "fire_0"),
    ("POLISI", "POLICE", "pol_"),
    ("AMBULANCE", "AMBULANCE", "amb_new"),
    ("DAMKAR (FIRETRUCK)", "FIRETRUCK", "fire_0"),
    ("POLISI", "POLICE", "pol_"),
    ("POLISI", "POLICE", "pol_")
]

# Cari path file asli berdasarkan kriteria di atas
selected_files = []
for target_cls, folder, pattern in TEST_SAMPLES:
    search_dir = os.path.join(ROOT, folder)
    matching_files = [f for f in glob.glob(os.path.join(search_dir, "*.wav")) if pattern in os.path.basename(f)]
    if not matching_files:
        matching_files = glob.glob(os.path.join(search_dir, "*.wav"))
    
    # Pilih 1 file secara acak dari yang cocok, pastikan tidak duplikat di daftar
    found = False
    for f in matching_files:
        if f not in selected_files:
            selected_files.append((target_cls, f))
            found = True
            break
    if not found and matching_files:
        selected_files.append((target_cls, matching_files[0]))

print("=" * 70)
print("     UJI INTERAKTIF 10 SUARA SIRINE REAL-TIME SIRENMASTER (8 DETIK)")
print("=" * 70)
print("[*] Dekatkan mic alat ESP32 Anda ke speaker komputer Anda!")
print("[*] Setiap sirene akan diputar 2x berturut-turut (total 8 detik).")
print("[*] Tekan ENTER setelah setiap tes untuk lanjut ke suara berikutnya.")
print("=" * 70)

input("[*] Siap? Tekan ENTER untuk memulai Tes #1...")

for idx, (target_cls, filepath) in enumerate(selected_files, 1):
    print(f"\n[📝 TES #{idx} / 10]")
    print(f"  • Target Deteksi Alat : {target_cls}")
    print(f"  • File Audio Diputar  : {os.path.basename(filepath)}")
    print("  • Memulai pemutaran dalam: 3... 2... 1...")
    time.sleep(1.0)
    
    # Putar 2 kali berturut-turut agar total pemutaran = 8 detik (1 file = 4 detik)
    print("  🔊 MEMBUNYIKAN SIRINE (Putaran 1/2)...")
    winsound.PlaySound(filepath, winsound.SND_FILENAME)
    
    print("  🔊 MEMBUNYIKAN SIRINE (Putaran 2/2)...")
    winsound.PlaySound(filepath, winsound.SND_FILENAME)
    
    print("  [+] Pemutaran selesai. Menunggu alat kembali ke status 'AMAN'...")
    time.sleep(4.0) # Jeda agar motor berhenti & LCD kembali AMAN
    
    if idx < 10:
        input(f"[*] Tekan ENTER untuk lanjut ke Tes #{idx+1}...")

print("\n" + "=" * 70)
print("  🏆 PENGUJIAN 10 SUARA SIRINE SELESAI!")
print("  Silakan catat hasil deteksi alat Anda untuk lampiran Skripsi Anda.")
print("=" * 70)
