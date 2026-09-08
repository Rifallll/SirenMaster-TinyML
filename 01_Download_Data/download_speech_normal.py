"""
download_speech_normal.py
=========================
Mengunduh video suara manusia berbicara (podcast & berita) menggunakan yt-dlp & ffmpeg.
Memotong suara tersebut menjadi chunk 4 detik dan menyimpannya di folder NORMAL (class 2)
agar model AI terbiasa mengenali suara manusia / obrolan sebagai NORMAL (bukan sirine).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, subprocess, glob, numpy as np, librosa, soundfile as sf

OUT_DIR = r"C:\Users\ASUS\Videos\DATASET\NORMAL"
os.makedirs(OUT_DIR, exist_ok=True)

# Hapus file speech_neg_*.wav lama jika ada
for old_f in glob.glob(os.path.join(OUT_DIR, "speech_neg_*.wav")):
    try: os.remove(old_f)
    except: pass

URLS = {
    "podcast_indo": "ytsearch1:podcast indonesia bicara santai wawancara",
    "berita_tv": "ytsearch1:berita tv indonesia pembawa acara bicara",
}

print("=" * 80)
print(" DOWNLOADING HUMAN SPEECH / TALK FOR NORMAL CLASS")
print("=" * 80)

for name, query in URLS.items():
    print(f"\n[*] Downloading {name}...")
    out_temp = os.path.join(OUT_DIR, f"temp_{name}.wav")
    
    # Hapus file temp lama jika ada
    if os.path.exists(out_temp):
        try: os.remove(out_temp)
        except: pass
        
    cmd = [
        "python", "-m", "yt_dlp",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "-ar 8000 -ac 1",
        "--download-sections", "*00:00:30-00:04:30", # Download 4 menit dari detik ke-30
        "-o", out_temp,
        query
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERR] Gagal mendownload {name}: {e}")
        continue

# Potong audio menjadi segmen 4 detik
temp_files = glob.glob(os.path.join(OUT_DIR, "temp_*.wav"))
total_chunks = 0

for tf in temp_files:
    basename = os.path.basename(tf).replace(".wav", "")
    print(f"\n[*] Processing chunks for {tf}...")
    try:
        y, sr = librosa.load(tf, sr=8000)
        chunk_samples = int(4.0 * sr) # 4 detik
        
        saved = 0
        for start in range(0, len(y) - chunk_samples, chunk_samples):
            chunk = y[start : start + chunk_samples]
            # Validasi tingkat volume (abaikan hening total)
            rms = np.sqrt(np.mean(chunk**2))
            if rms < 0.005:
                continue
                
            out_path = os.path.join(OUT_DIR, f"speech_neg_{basename}_part{saved:04d}.wav")
            sf.write(out_path, chunk, sr)
            saved += 1
            total_chunks += 1
            
        print(f"  -> Berhasil membuat {saved} file segmen 4 detik.")
        os.remove(tf) # Hapus file mentah panjang
    except Exception as e:
        print(f"[ERR] Gagal memotong file {tf}: {e}")

print("\n" + "=" * 80)
print(f"[SUCCESS] Selesai! Berhasil menambahkan {total_chunks} segmen percakapan/podcast ke NORMAL.")
print("=" * 80)
