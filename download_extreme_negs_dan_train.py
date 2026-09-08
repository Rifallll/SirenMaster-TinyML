"""
download_extreme_negs_dan_train.py
===========================
1. Download suara-suara ekstrem level dewa (Feedback mic, rem kereta, blender, suling, dll)
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
DOWNLOAD_DIR = r"C:\Users\ASUS\Videos\DATASET\TRASH\extremenegs_download"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(NORMAL_DIR, exist_ok=True)

# ── Daftar pencarian YouTube untuk suara-suara Ujian Level Dewa ──
SEARCH_QUERIES = [
    "ytsearch1:suara microphone feedback dengung",
    "ytsearch1:suara pengereman kereta api decit",
    "ytsearch1:suara mesin blender halus",
    "ytsearch1:suara tiupan suling bambu panjang",
    "ytsearch1:suara ban mobil selip nge-drift",
    "ytsearch1:suara kembang api meluncur peluit",
    "ytsearch1:suara orang bersiul nyaring"
]

print("="*65)
print(" DOWNLOADER SUARA PENGECOH LEVEL DEWA (EXTREME ROAD TEST)")
print("="*65)
print(f" Target: {len(SEARCH_QUERIES)} jenis suara → dipotong jadi segmen 4 detik")
print()

downloaded = []
for i, query in enumerate(SEARCH_QUERIES):
    print(f"[{i+1}/{len(SEARCH_QUERIES)}] Mencari & Mendownload: {query.replace('ytsearch1:', '')}")
    out_tpl = os.path.join(DOWNLOAD_DIR, f"extremeneg_{i+1:02d}.%(ext)s")
    result = subprocess.run([
        sys.executable, "-m", "yt_dlp",
        "-x",                     
        "--audio-format", "mp3",  
        "--audio-quality", "0",   
        "-o", out_tpl,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        query
    ], capture_output=True, text=True)
    
    # Cari file yang terdownload
    found = glob.glob(os.path.join(DOWNLOAD_DIR, f"extremeneg_{i+1:02d}.*"))
    if found:
        print(f"    ✅ OK -> {os.path.basename(found[0])}")
        downloaded.append(found[0])
    else:
        print(f"    ⚠️  SKIP (gagal download): {result.stderr[:80] if result.stderr else 'unknown error'}")

print(f"\n Berhasil download: {len(downloaded)} dari {len(SEARCH_QUERIES)} jenis suara\n")

print("="*65)
print(" Memotong suara jadi segmen 4 detik @ 8kHz mono...")
print("="*65)

total_segments = 0
ts = int(time.time())

for fpath in downloaded:
    print(f"\n[*] Memproses: {os.path.basename(fpath)}")
    try:
        y, sr = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)
        durasi = len(y) / SAMPLE_RATE
        n_seg  = int(durasi // DURATION)
        print(f"    Durasi: {durasi:.1f} detik -> {n_seg} segmen")
        
        base = os.path.splitext(os.path.basename(fpath))[0]
        for s in range(n_seg):
            segment = y[s*TARGET_LEN:(s+1)*TARGET_LEN]
            if len(segment) < TARGET_LEN:
                continue
            mx = np.max(np.abs(segment))
            if mx > 1e-6:
                segment = segment * (0.88 / mx) # Normalize volume
            out_name = os.path.join(NORMAL_DIR, f"extreme_noise_{base}_seg{s+1:04d}_{ts}.wav")
            out_data = np.clip(segment * 32767, -32768, 32767).astype(np.int16)
            wav.write(out_name, SAMPLE_RATE, out_data)
            total_segments += 1
        
        print(f"    ✅ {n_seg} segmen tersimpan ke NORMAL!")
    except Exception as e:
        print(f"    ❌ Error: {e}")

print(f"\n Total segmen suara pengecoh level dewa: {total_segments} file di folder NORMAL")

if total_segments > 0:
    print("\n" + "="*65)
    print(" Mulai Retrain Model AI (Memasukkan data pengecoh EXTREME)...")
    print("="*65)
    train_script = r"C:\Users\ASUS\Videos\DATASET\04_Training_AI\train_lokal.py"
    subprocess.run(["python", train_script], cwd=r"C:\Users\ASUS\Videos\DATASET\04_Training_AI")
    print("\n ✅ SELESAI! Model SUPER ULTIMATE siap.")
