"""
download_telolet.py
===================
Mengunduh suara klakson telolet Bus Basuri (melodi panjang & melengking)
dari YouTube, lalu memotongnya menjadi potongan 4 detik untuk folder NORMAL.
Ini mencegah AI salah mendeteksi melodi klakson telolet sebagai sirine darurat.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, subprocess, glob, librosa, numpy as np, soundfile as sf

OUT_DIR = r"C:\Users\ASUS\Videos\DATASET\NORMAL"
os.makedirs(OUT_DIR, exist_ok=True)

URLS = {
    "telolet_basuri_v3":        "ytsearch1:suara klakson telolet basuri v3 asli panjang",
    "telolet_basuri_mengular":  "ytsearch1:klakson telolet basuri mengular premium sound",
    "telolet_basuri_alzifa":    "ytsearch1:telolet basuri alzifa pianika sound effect",
    "telolet_basuri_melodi":    "ytsearch1:suara telolet basuri corong 6 melodi panjang",
}

print("=" * 65)
print("  DOWNLOAD KLAKSON TELOLET BASURI UNTUK KELAS NORMAL")
print("=" * 65)

for name, url in URLS.items():
    print(f"\n[>] Mengunduh: {name} ...")
    out_template = os.path.join(OUT_DIR, f"real_telolet_{name}_%(id)s.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "0",
        "--download-sections", "*00:00:00-00:02:00", # Ambil 2 menit saja per video
        "--no-playlist",
        "-o", out_template,
        url
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"  [!] Gagal: {e}")

print("\n[*] Memotong audio telolet menjadi 4 detik...")
mp3_files = glob.glob(os.path.join(OUT_DIR, "real_telolet_*.mp3"))

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
            # Lewati jika bagian audio tersebut hening/senyap
            if float(np.sqrt(np.mean(chunk**2))) < 0.005:
                continue
            sf.write(os.path.splitext(wf)[0] + f"_part{saved:04d}.wav", chunk, sr)
            saved += 1
            total += 1
        print(f"     [OK] {saved} sampel.")
        os.remove(wf)
    except Exception as e:
        print(f"     [!] Error: {e}")

for f in glob.glob(os.path.join(OUT_DIR, "real_telolet_*.mp3")):
    try: os.remove(f)
    except: pass

print("\n" + "=" * 65)
print(f"  [SUCCESS] {total} sampel klakson telolet Basuri berhasil masuk ke NORMAL!")
print("=" * 65)
