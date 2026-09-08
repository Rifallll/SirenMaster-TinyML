"""
play_siren.py
=============
Memutar file audio suara Damkar (Firetruck) mekanik asli dari dataset
melalui speaker komputer Anda agar mic alat ESP32 Anda bisa mendengarnya
dan mendeteksinya secara langsung sebagai DAMKAR.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, time, winsound

ROOT = r"C:\Users\ASUS\Videos\DATASET"
# Pilih file suara Damkar mekanik asli dari dataset (fire_0002_seg01.wav)
AUDIO_FILE = os.path.join(ROOT, "FIRETRUCK", "fire_0002_seg01.wav")

if not os.path.exists(AUDIO_FILE):
    # Fallback ke file wave damkar lainnya jika file di atas tidak sengaja terhapus
    files = [f for f in os.listdir(os.path.join(ROOT, "FIRETRUCK")) if f.endswith(".wav")]
    if files:
        AUDIO_FILE = os.path.join(ROOT, "FIRETRUCK", files[0])

print("=" * 65)
print("  PENGUJIAN SUARA SIRINE DAMKAR MEKANIK (FEDERAL SIGNAL Q2B)")
print("=" * 65)
print(f"[*] File audio: {os.path.basename(AUDIO_FILE)}")
print("[*] Dekatkan alat ESP32 Anda ke speaker laptop/komputer Anda!")
print("[*] Siap membunyikan dalam 3 detik...")
time.sleep(1.0)
print("  -> 2...")
time.sleep(1.0)
print("  -> 1...")
time.sleep(1.0)

print("\n🔊 MEMBUNYIKAN SUARA SIRINE...")
# winsound.PlaySound memutar WAV secara sinkronus di Windows
winsound.PlaySound(AUDIO_FILE, winsound.SND_FILENAME)

print("\n[+] Selesai memutar!")
print("=============================================================")
print("Perhatikan layar LCD alat Anda. Layar harus mendeteksi")
print("warna ORANYE dengan nama 'DAMKAR' saat suara tadi diputar! 🚒")
print("=============================================================")
