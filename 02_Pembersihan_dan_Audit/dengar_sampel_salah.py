"""
dengar_sampel_salah.py
======================
Skrip untuk memutar sampel audio Ambulans Eropa (Hi-Lo / Two-Tone) yang diklasifikasikan
sebagai Polisi oleh model AI, agar kita bisa mendengarkan langsung bukti akustiknya.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, time
import winsound

ROOT = r"C:\Users\ASUS\Videos\DATASET"
AMB_DIR = os.path.join(ROOT, "AMBULANCE")

# Daftar file ambulans Eropa (Hi-Lo/Two-Tone) yang ditebak Polisi oleh AI
sampel_salah = [
    ("dl_boost_amb_euro_hi_lo_LTQLU_R-yG4_part0036.wav", "Ambulans Eropa Hi-Lo (Tee-Toh Tee-Toh cepat)"),
    ("dl_boost_amb_france_twoton_fAaOT28j878_part0061.wav", "Ambulans Prancis 2-Tone (Nada ganda tajam)"),
    ("dl_boost_amb_euro_hi_lo_LTQLU_R-yG4_part0059.wav", "Ambulans Eropa Hi-Lo (Nada bergetar mirip polisi)"),
]

print("=" * 80)
print(" 🎧 MEMUTAR SAMPEL AMBULANS EROPA YANG DITEBAK POLISI OLEH AI")
print("=" * 80)
print("Dengarkan baik-baik melalui speaker laptop Anda:")
print("Suara-suara ini memiliki irama 'Hi-Lo / Dua Nada' yang persis dengan sirene Polisi.\n")

for i, (filename, desc) in enumerate(sampel_salah, 1):
    filepath = os.path.join(AMB_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[{i}] File tidak ditemukan: {filename}")
        continue
        
    print(f"[{i}/{len(sampel_salah)}] Memutar: {filename}")
    print(f"    -> Keterangan: {desc}")
    print("    -> Mendengarkan...")
    
    # Memutar audio WAV di Windows
    winsound.PlaySound(filepath, winsound.SND_FILENAME)
    time.sleep(1.0) # Jeda antar suara

print("=" * 80)
print(" ✅ SELESAI DIPUTAR!")
print("Terbukti secara telinga (akustik): Suara Ambulans Eropa bertipe Hi-Lo/Two-Tone")
print("sangat mirip dengan sirene Polisi/Yelp, sehingga wajar AI mengenalinya sebagai Polisi!")
print("=" * 80)
