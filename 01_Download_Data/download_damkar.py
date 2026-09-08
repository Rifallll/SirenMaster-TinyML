"""Unduh tambahan FIRETRUCK khusus - durasi longgar sampai 20 menit."""
import sys, os, subprocess
import librosa, numpy as np
from scipy.io import wavfile
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUT_DIR     = r"DATASET_LOKAL\FIRETRUCK"
SAMPLE_RATE = 8000
CHUNK       = int(8000 * 1.024)
RMS_THR     = 0.05

QUERIES = [
    "ytsearch1:suara sirine damkar pemadam kebakaran indonesia terbaru",
    "ytsearch1:suara sirine pemadam kebakaran keras lantang",
    "ytsearch1:sound fire truck siren indonesia long",
    "ytsearch1:suara sirine damkar dki buka jalan kecelakaan",
    "ytsearch1:klakson airhorn pemadam kebakaran truk",
    "ytsearch1:firetruck siren sound effect indonesia",
    "ytsearch1:suara truk pemadam lewat kencang",
    "ytsearch1:sirine damkar bali surabaya bandung",
]

os.makedirs(OUT_DIR, exist_ok=True)
total_saved = 0

for i, q in enumerate(QUERIES, 1):
    tmp = f"_tmp_dam_{i}.mp3"
    print(f"[{i:02d}/{len(QUERIES)}] {q[:70]}")
    cmd = [
        "python", "-m", "yt_dlp",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "0",
        "--match-filter", "duration < 1800",   # Longgar hingga 30 menit
        "--no-playlist",
        "-o", tmp, q
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception:
        pass

    if not os.path.exists(tmp):
        print("    [SKIP]")
        continue

    try:
        y, _ = librosa.load(tmp, sr=SAMPLE_RATE, mono=True)
        saved = 0
        for j in range(len(y) // CHUNK):
            chunk = y[j*CHUNK:(j+1)*CHUNK]
            if np.sqrt(np.mean(chunk**2)) < RMS_THR:
                continue
            out = os.path.join(OUT_DIR, f"dam_extra_{i}_{j:04d}.wav")
            wavfile.write(out, SAMPLE_RATE, np.int16(chunk * 32767))
            saved += 1
        total_saved += saved
        print(f"    [OK] {saved} cuplikan")
        os.remove(tmp)
    except Exception as e:
        print(f"    [ERROR] {e}")

total = len([f for f in os.listdir(OUT_DIR) if f.endswith('.wav')])
print(f"\nTOTAL FIRETRUCK: {total} file ({total_saved} baru)")
