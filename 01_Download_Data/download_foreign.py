import os
import subprocess

searches = [
    # POLICE
    ("POLICE", "NYPD police siren sound effect clear"),
    ("POLICE", "European police siren sound effect uk london"),
    ("POLICE", "French police siren sound effect"),
    # AMBULANCE
    ("AMBULANCE", "London ambulance siren sound effect clear"),
    ("AMBULANCE", "FDNY ambulance siren sound effect"),
    ("AMBULANCE", "German ambulance siren sound effect"),
    # FIRETRUCK
    ("FIRETRUCK", "FDNY fire truck siren horn sound effect clear"),
    ("FIRETRUCK", "European fire engine siren sound effect"),
    ("FIRETRUCK", "Japanese fire truck siren sound effect")
]

SAVE_DIR = "TAMBAH_DATASET"
os.makedirs(SAVE_DIR, exist_ok=True)

print("[*] Memulai perburuan sirine luar negeri...")

for category, query in searches:
    print(f"\n[+] Mencari {category}: '{query}'...")
    cat_dir = os.path.join(SAVE_DIR, category)
    os.makedirs(cat_dir, exist_ok=True)
    
    # Download top 2 results for each query, extract audio, max 30 seconds duration
    cmd = [
        "python", "-m", "yt_dlp",
        f"ytsearch2:{query}",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "-ac 1 -ar 8000 -t 30", # 1 channel, 8000Hz, max 30 seconds
        "-o", f"{cat_dir}/foreign_%(id)s.%(ext)s"
    ]
    subprocess.run(cmd)

print("\n[SUCCESS] Semua sirine luar negeri berhasil didownload dan dimasukkan ke TAMBAH_DATASET!")
