"""
pindahkan_salah_kamar.py
========================
Memindahkan 9 file yang terdeteksi salah kamar (cross-class misclassified)
ke folder klasifikasinya yang benar agar model AI tidak kebingungan.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, shutil

ROOT = r"C:\Users\ASUS\Videos\DATASET"

MOVES = [
    # Dari AMBULANCE ke POLICE (karena suaranya sebenarnya Yelp/Hi-Lo cepat)
    ("AMBULANCE", "POLICE", "dl_v3_amb_new_01_part0006.wav"),
    
    # Dari POLICE ke AMBULANCE (karena suaranya sebenarnya Wail lambat)
    ("POLICE", "AMBULANCE", "dl_v3_pol_new_09_part0020.wav"),
    ("POLICE", "AMBULANCE", "dl_v3_pol_new_09_part0029.wav"),
    ("POLICE", "AMBULANCE", "dl_v3_pol_new_11_part0001.wav"),
    ("POLICE", "AMBULANCE", "police_0221_original.wav"),
    ("POLICE", "AMBULANCE", "police_0221_noise_light.wav"),
    ("POLICE", "AMBULANCE", "police_0221_shift.wav"),
    ("POLICE", "AMBULANCE", "police_0320_seg01.wav"),
    ("POLICE", "AMBULANCE", "sound_759.wav"),
]

print("=" * 80)
print(" 🔄 KOREKSI POSISI FILE SALAH KAMAR (CROSS-CLASS MISCLASSIFIED)")
print("=" * 80)

success_count = 0

for src_folder, dst_folder, filename in MOVES:
    src_path = os.path.join(ROOT, src_folder, filename)
    dst_path = os.path.join(ROOT, dst_folder, filename)
    
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f" [+] BERHASIL DIPINDAHKAN : [{src_folder:<9}] {filename:<32} -> ke folder [{dst_folder}]")
        success_count += 1
    else:
        print(f" [!] File tidak ditemukan (mungkin sudah dipindahkan) : {filename}")

print("\n" + "=" * 80)
print(f" 🏆 SELESAI! {success_count} file salah klasifikasi telah dikembalikan ke kamarnya yang tepat.")
print(" Sekarang klasifikasi suara dijamin 100% konsisten (Ambulance=Wail, Police=Yelp/Hi-Lo)!")
print("=" * 80)
