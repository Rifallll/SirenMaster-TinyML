"""
download_music_dan_train.py
===========================
1. Download lagu rock/metal/pop royalty-free dari YouTube
2. Potong jadi segmen 4 detik @ 8kHz mono
3. Simpan ke folder NORMAL
4. Retrain model AI
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, time, subprocess, glob
import numpy as np
import librosa
import scipy.io.wavfile as wav

SAMPLE_RATE  = 8000
DURATION     = 4.0
TARGET_LEN   = int(SAMPLE_RATE * DURATION)
NORMAL_DIR   = r"C:\Users\ASUS\Videos\DATASET\NORMAL"
DOWNLOAD_DIR = r"C:\Users\ASUS\Videos\DATASET\TRASH\music_download"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(NORMAL_DIR, exist_ok=True)

# ── Daftar lagu royalty-free (NoCopyrightSounds, YouTube Audio Library, dll) ──
TRACK_LIST = [
    # Rock / Metal NCS
    "https://www.youtube.com/watch?v=6Om0b9pVLFM",  # Elektronomia - Sky High
    "https://www.youtube.com/watch?v=yJg-Y5byMMw",  # Jim Yosef - Link
    "https://www.youtube.com/watch?v=qlMpBQMWKYA",  # Warriyo - Mortals
    "https://www.youtube.com/watch?v=TgFqMFfdYPo",  # Different Heaven - Nekozilla
    "https://www.youtube.com/watch?v=2Nv5juZKhKo",  # Spectacles Wallet & Watch
    "https://www.youtube.com/watch?v=Fa3vTKoNFJw",  # K-391 - Earth
    "https://www.youtube.com/watch?v=S2fXS8Vb9fE",  # elektronomia - energy
    "https://www.youtube.com/watch?v=dSw7fQqzBIU",  # JJD - Future
    "https://www.youtube.com/watch?v=8GW6sLrK40k",  # NCS: Rock Mix
    "https://www.youtube.com/watch?v=h9tRmVpNAik",  # Elektronomia - Limitless
]

print("="*65)
print(" DOWNLOADER MUSIK ROCK/EDM ROYALTY-FREE")
print(" Sumber: NoCopyrightSounds & YouTube Audio Library")
print("="*65)
print(f" Target: {len(TRACK_LIST)} lagu → dipotong jadi segmen 4 detik")
print()

# ── STEP 1: Download ──────────────────────────────────────────────────
downloaded = []
for i, url in enumerate(TRACK_LIST):
    print(f"[{i+1}/{len(TRACK_LIST)}] Downloading: {url}")
    out_tpl = os.path.join(DOWNLOAD_DIR, f"rock_{i+1:02d}.%(ext)s")
    result = subprocess.run([
        sys.executable, "-m", "yt_dlp",
        "-x",                     # extract audio only
        "--audio-format", "mp3",  # format mp3
        "--audio-quality", "0",   # kualitas terbaik
        "-o", out_tpl,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url
    ], capture_output=True, text=True)
    
    outfile = os.path.join(DOWNLOAD_DIR, f"rock_{i+1:02d}.mp3")
    if os.path.exists(outfile):
        print(f"    ✅ OK -> {os.path.basename(outfile)}")
        downloaded.append(outfile)
    else:
        # Coba cari file apapun yang dihasilkan
        found = glob.glob(os.path.join(DOWNLOAD_DIR, f"rock_{i+1:02d}.*"))
        if found:
            print(f"    ✅ OK -> {os.path.basename(found[0])}")
            downloaded.append(found[0])
        else:
            print(f"    ⚠️  SKIP (gagal download): {result.stderr[:80] if result.stderr else 'unknown error'}")

print(f"\n Berhasil download: {len(downloaded)} dari {len(TRACK_LIST)} lagu\n")

# ── STEP 2: Potong jadi segmen 4 detik ───────────────────────────────
print("="*65)
print(" STEP 2: Memotong lagu jadi segmen 4 detik @ 8kHz mono...")
print("="*65)

total_segments = 0
ts = int(time.time())

for fpath in downloaded:
    print(f"\n[*] Memproses: {os.path.basename(fpath)}")
    try:
        # Load audio dengan librosa (otomatis handle mp3, m4a, dll)
        y, sr = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)
        durasi = len(y) / SAMPLE_RATE
        n_seg  = int(durasi // DURATION)
        print(f"    Durasi: {durasi:.1f} detik -> {n_seg} segmen")
        
        base = os.path.splitext(os.path.basename(fpath))[0]
        for s in range(n_seg):
            segment = y[s*TARGET_LEN:(s+1)*TARGET_LEN]
            if len(segment) < TARGET_LEN:
                continue
            # Normalisasi volume
            mx = np.max(np.abs(segment))
            if mx > 1e-6:
                segment = segment * (0.88 / mx)
            # Simpan ke NORMAL
            out_name = os.path.join(NORMAL_DIR, f"music_{base}_seg{s+1:04d}_{ts}.wav")
            out_data = np.clip(segment * 32767, -32768, 32767).astype(np.int16)
            wav.write(out_name, SAMPLE_RATE, out_data)
            total_segments += 1
        
        print(f"    ✅ {n_seg} segmen tersimpan ke NORMAL!")
    except Exception as e:
        print(f"    ❌ Error: {e}")

print(f"\n Total segmen musik baru: {total_segments} file di folder NORMAL")

if total_segments == 0:
    print("[!] Tidak ada segmen yang dihasilkan. Keluar.")
    sys.exit(1)

# ── STEP 3: Retrain ───────────────────────────────────────────────────
print("\n" + "="*65)
print(" STEP 3: Retrain Model AI dengan Data Musik Baru...")
print(" (Estimasi: 30-40 menit, harap tunggu!)")
print("="*65)

train_script = r"C:\Users\ASUS\Videos\DATASET\04_Training_AI\train_lokal.py"
result = subprocess.run(
    ["python", train_script],
    cwd=r"C:\Users\ASUS\Videos\DATASET\04_Training_AI"
)

if result.returncode == 0:
    print("\n" + "="*65)
    print(" ✅ SELESAI! Model baru: sirenmaster_main/model.h")
    print(" Upload ke ESP32 — musik rock tidak akan deteksi lagi!")
    print("="*65)
else:
    print("\n[!] Training gagal. Periksa error di atas.")
