"""
download_motor_sounds.py
========================
Download suara motor sport, matic, dan variasi motor jalanan Indonesia
untuk memperkuat kelas NORMAL agar tidak disangka sirine oleh AI.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, subprocess, glob, librosa, numpy as np, soundfile as sf

OUT_DIR = r"C:\Users\ASUS\Videos\DATASET\NORMAL"
os.makedirs(OUT_DIR, exist_ok=True)

URLS = {
    # Motor Sport / Racing - Frekuensi Tinggi (paling berbahaya utk false alarm!)
    "motorsport_cbr_revving":   "ytsearch1:honda CBR 600 motorcycle revving sound effect high rpm",
    "motorsport_ninja_racing":  "ytsearch1:kawasaki ninja motorcycle acceleration high rpm sound",
    "motorsport_r15_revving":   "ytsearch1:yamaha r15 motorcycle revving exhaust sound loud",

    # Motor Matic Indonesia
    "motor_matic_vario":        "ytsearch1:suara motor matic honda vario akselerasi jalan raya",
    "motor_matic_nmax":         "ytsearch1:suara motor yamaha nmax jalan raya indonesia",
    "motor_beat_jalanan":       "ytsearch1:suara motor beat jalan raya kota indonesia",

    # Kondisi Spesifik Motor di Jalan
    "motor_mogok_starter":      "ytsearch1:suara motor susah distarter mogok berulang",
    "motor_klakson_ramai":      "ytsearch1:suara klakson motor banyak ramai jalanan indonesia",
    "motor_trek_liar":          "ytsearch1:suara balapan motor trek liar jalanan malam",
}

print("=" * 65)
print("  DOWNLOAD SUARA MOTOR JALANAN UNTUK KELAS NORMAL")
print("=" * 65)

for name, url in URLS.items():
    print(f"\n[>] Mengunduh: {name} ...")
    out_template = os.path.join(OUT_DIR, f"real_motor_{name}_%(id)s.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "0",
        "--download-sections", "*00:00:00-00:03:00",
        "--no-playlist",
        "-o", out_template,
        url
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"  [!] Gagal: {e}")

print("\n[*] Memotong audio menjadi 4 detik...")
mp3_files = glob.glob(os.path.join(OUT_DIR, "real_motor_*.mp3"))

total = 0
for wf in mp3_files:
    if "_part" in wf:
        continue
    print(f"  -> {os.path.basename(wf)} ...")
    try:
        y, sr = librosa.load(wf, sr=8000)
        chunk_len = int(4.0 * sr)
        saved = 0
        for i in range(0, len(y) - chunk_len, chunk_len):
            chunk = y[i:i+chunk_len]
            if float(np.sqrt(np.mean(chunk**2))) < 0.001:
                continue
            sf.write(os.path.splitext(wf)[0] + f"_part{saved:04d}.wav", chunk, sr)
            saved += 1
            total += 1
        print(f"     [OK] {saved} sampel.")
        os.remove(wf)
    except Exception as e:
        print(f"     [!] Error: {e}")

for f in glob.glob(os.path.join(OUT_DIR, "real_motor_*.mp3")):
    try: os.remove(f)
    except: pass

print("\n" + "=" * 65)
print(f"  [SUCCESS] {total} sampel suara motor baru ditambahkan ke NORMAL!")
print("=" * 65)
