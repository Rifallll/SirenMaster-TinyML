import os
import subprocess
import sys
import glob

import imageio_ffmpeg
import librosa
import soundfile as sf
import numpy as np

# Use ffmpeg installed via imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
print(f"Using ffmpeg from: {ffmpeg_exe}")

# Using ytsearch to bypass direct link blocks
URLS = {
    # Suara outdoor (sudah ada sebelumnya)
    "pasar": "ytsearch1:suara pasar tradisional asmr",
    "mall": "ytsearch1:suara keramaian mall asmr",
    "jalanan_klakson": "ytsearch1:suara lalu lintas jalan raya klakson Indonesia",

    # === HARD NEGATIVE BARU: Suara dalam ruangan & kehidupan sehari-hari ===
    # Ini yang menyebabkan false positive POLICE saat ini!
    "obrolan": "ytsearch1:suara orang ngobrol ramai kafe Indonesia",
    "musik_jalanan": "ytsearch1:suara musik jalanan pedagang asongan Indonesia",
    "motor_bus": "ytsearch1:suara kemacetan motor bus jalanan Indonesia",
    "warung": "ytsearch1:suara warung makan ramai ambience Indonesia",
    "murottal": "ytsearch1:suara masjid adzan dan keramaian Indonesia ambient",
    "kipas_ac": "ytsearch1:white noise kipas angin ruangan",
}

OUT_DIR = r"c:\Users\ASUS\Videos\DATASET\NORMAL"
os.makedirs(OUT_DIR, exist_ok=True)

for name, url in URLS.items():
    print(f"--- Downloading {name} ---")
    out_template = os.path.join(OUT_DIR, f"hard_neg_{name}_%(id)s.%(ext)s")
    
    # We download ONLY the first 3 minutes of the video to prevent downloading 10-hour videos
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--ffmpeg-location", ffmpeg_exe,
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "-ar 8000 -ac 1",
        "--download-sections", "*00:00:00-00:03:00", # ONLY FIRST 3 MINS
        "-o", out_template,
        url
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {name}: {e}")

print("--- Chunking downloaded audio ---")
wav_files = glob.glob(os.path.join(OUT_DIR, "hard_neg_*.wav"))

for wf in wav_files:
    if "_part" in wf:
        continue
    print(f"Processing {wf}...")
    try:
        y, sr = librosa.load(wf, sr=8000)
        chunk_len = int(4.0 * sr) # 4 seconds
        
        saved = 0
        for i in range(0, len(y) - chunk_len, chunk_len):
            chunk = y[i:i+chunk_len]
            rms = np.sqrt(np.mean(chunk**2))
            if rms < 0.01:
                continue
            
            out_name = wf.replace(".wav", f"_part{saved:04d}.wav")
            sf.write(out_name, chunk, sr)
            saved += 1
            
        print(f"  -> Generated {saved} chunks of 4-seconds.")
        os.remove(wf) # Remove original long file
    except Exception as e:
        print(f"Error chunking {wf}: {e}")

print("Done! Added hard negatives to NORMAL folder. Now you can run train_lokal.py.")
