import os
import subprocess
import librosa
import numpy as np
from scipy.io import wavfile

# Konfigurasi
OUT_DIR = "DATASET_LOKAL"
os.makedirs(OUT_DIR, exist_ok=True)

# Kueri pencarian YouTube spesifik agar murni suara sirine
CLASSES = {
    "POLISI": [
        "ytsearch1:suara sirine patwal polisi indonesia murni",
        "ytsearch1:suara sirine polisi militer indonesia"
    ],
    "PEMADAM": [
        "ytsearch1:suara sirine damkar indonesia keras",
        "ytsearch1:suara sirine pemadam kebakaran indonesia tanpa jeda"
    ],
    "AMBULANS": [
        "ytsearch1:suara sirine ambulans ninu ninu indonesia",
        "ytsearch1:suara sirine ambulans gawat darurat indonesia"
    ]
}

CHUNK_LEN_S = 1.0
SAMPLE_RATE = 8000
RMS_THRESHOLD = 0.07  # Cukup tinggi agar suara ngobrol/kresek tidak ikut masuk

for cls, queries in CLASSES.items():
    cls_dir = os.path.join(OUT_DIR, cls)
    os.makedirs(cls_dir, exist_ok=True)
    
    for i, q in enumerate(queries):
        print(f"\n[{cls}] Mengunduh video {i+1}...")
        tmp_mp3 = f"tmp_{cls}_{i}.mp3"
        
        # Download menggunakan yt-dlp
        cmd = [
            "python", "-m", "yt_dlp", 
            "-x", "--audio-format", "mp3",
            "--match-filter", "duration < 240", # Maksimal 4 menit untuk menghindari vlog
            "-o", tmp_mp3,
            q
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Gagal unduh, lanjut ke kueri berikutnya...")
            continue
        
        if not os.path.exists(tmp_mp3):
            continue
            
        print(f"[{cls}] Memotong audio dan membuang yang hening/bukan sirine...")
        try:
            y, sr = librosa.load(tmp_mp3, sr=SAMPLE_RATE, mono=True)
            chunk_samples = int(SAMPLE_RATE * CHUNK_LEN_S)
            num_chunks = len(y) // chunk_samples
            
            saved_count = 0
            for j in range(num_chunks):
                start = j * chunk_samples
                end = start + chunk_samples
                chunk = y[start:end]
                
                # Cek kekerasan suara (RMS)
                rms = np.sqrt(np.mean(chunk**2))
                if rms > RMS_THRESHOLD:
                    # Simpan sebagai WAV 16-bit PCM
                    out_path = os.path.join(cls_dir, f"yt_indo_{i}_part_{j}.wav")
                    chunk_int16 = np.int16(chunk * 32767)
                    wavfile.write(out_path, SAMPLE_RATE, chunk_int16)
                    saved_count += 1
                    
            print(f"[{cls}] Berhasil menyimpan {saved_count} cuplikan sirine murni.")
            os.remove(tmp_mp3) # Hapus file MP3 mentah
        except Exception as e:
            print(f"Gagal memproses audio: {e}")

print("\n" + "="*50)
print("PROSES UNDUH DAN POTONG SELESAI!")
print(f"File telah disimpan ke dalam folder: {OUT_DIR}")
print("="*50)
