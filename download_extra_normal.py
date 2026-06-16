"""
Script ini mengunduh suara NORMAL tambahan yang sangat spesifik untuk mengurangi false positive.
Fokus pada suara yang mirip dengan frekuensi sirine polisi (200-800Hz):
  - Suara manusia ngobrol (harmonik vokal)
  - Suara musik elektronik (synthesizer, bass)
  - Suara kendaraan berbeda (motor 2-tak, bajaj)
  - Suara alam (hujan, angin, burung)
"""
import os
import subprocess
import sys
import glob

import imageio_ffmpeg
import librosa
import soundfile as sf
import numpy as np

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
print(f"Using ffmpeg: {ffmpeg_exe}")

# === SUARA YANG SERING MENYEBABKAN FALSE POSITIVE ===
EXTRA_URLS = {
    # Suara manusia (harmonic vocal frequencies mirip sirine)
    "kerumunan_stadion":    "ytsearch1:suara penonton stadion sepakbola Indonesia ramai",
    "demo_orang":           "ytsearch1:suara kerumunan orang ramai demo jalan",
    "pasar_malam":          "ytsearch1:suara pasar malam keramaian Indonesia",

    # Suara kendaraan non-sirine
    "motor_2tak":           "ytsearch1:suara motor 2 tak klakson Jakarta",
    "bajaj":                "ytsearch1:suara bajaj angkot keramaian Jakarta",
    "kereta":               "ytsearch1:suara kereta KRL lewat stasiun Indonesia",
    "pesawat_lewat":        "ytsearch1:suara pesawat terbang lewat rendah",

    # Suara musik yang sering diputar di jalan
    "dangdut_speaker":      "ytsearch1:suara musik dangdut speaker outdoor jalanan",
    "speaker_toa_masjid":   "ytsearch1:suara pengumuman speaker masjid toa Indonesia",

    # Suara alam & lingkungan
    "hujan_lebat":          "ytsearch1:suara hujan lebat di jalanan Indonesia ambient",
    "angin_kencang":        "ytsearch1:suara angin kencang outdoor ambient",
    "petir_hujan":          "ytsearch1:suara petir dan hujan ambient ASMR",

    # Suara yang sangat mirip sirine tapi bukan (KRITIS!)
    "klakson_truk":         "ytsearch1:suara klakson truk besar container Indonesia",
    "alarm_motor":          "ytsearch1:suara alarm motor bunyi keras",
    "bel_sekolah":          "ytsearch1:suara bel sekolah",
}

OUT_DIR = r"c:\Users\ASUS\Videos\DATASET\NORMAL"
os.makedirs(OUT_DIR, exist_ok=True)

total = len(EXTRA_URLS)
for i, (name, url) in enumerate(EXTRA_URLS.items(), 1):
    print(f"\n[{i}/{total}] Downloading: {name}")
    out_template = os.path.join(OUT_DIR, f"extra_{name}_%(id)s.%(ext)s")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--ffmpeg-location", ffmpeg_exe,
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "-ar 8000 -ac 1",
        "--download-sections", "*00:00:00-00:03:00",
        "-o", out_template,
        url
    ]
    try:
        subprocess.run(cmd, check=True, timeout=120)
        print(f"  [OK] {name} downloaded.")
    except subprocess.TimeoutExpired:
        print(f"  [SKIP] {name} timeout.")
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] {name}: {e}")

print("\n--- Chunking semua file baru ---")
# Proses file extra_*.wav
wav_files = glob.glob(os.path.join(OUT_DIR, "extra_*.wav"))

total_chunks = 0
for wf in wav_files:
    if "_part" in wf:
        continue
    print(f"  Chunking {os.path.basename(wf)}...")
    try:
        y, sr = librosa.load(wf, sr=8000)
        chunk_len = int(4.0 * sr)

        saved = 0
        for i in range(0, len(y) - chunk_len, chunk_len):
            chunk = y[i:i+chunk_len]
            rms = np.sqrt(np.mean(chunk**2))
            if rms < 0.005:  # Lebih ketat: abaikan yang terlalu senyap
                continue

            out_name = wf.replace(".wav", f"_part{saved:04d}.wav")
            sf.write(out_name, chunk, sr)
            saved += 1

        total_chunks += saved
        print(f"    -> {saved} chunks dibuat.")
        os.remove(wf)
    except Exception as e:
        print(f"    [ERROR] {e}")

print(f"\n[SELESAI] Total {total_chunks} chunk baru ditambahkan ke folder NORMAL.")
print("Sekarang jalankan: .venv\\Scripts\\python.exe train_lokal.py")
