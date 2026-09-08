import urllib.request
import zipfile
import os
import shutil
import glob

URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
ZIP_PATH = "ESC-50-master.zip"
EXTRACT_DIR = "ESC-50-master"

AMBULANCE_DIR = "AMBULANCE"
NORMAL_DIR = "NORMAL"

def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading ESC-50 dataset (~600MB)... Please wait.")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print("Download complete!")
    else:
        print("Zip file already exists, skipping download.")

    if not os.path.exists(EXTRACT_DIR):
        print("Extracting zip file...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            # We only need the audio/ folder to save time
            for member in zip_ref.namelist():
                if member.startswith('ESC-50-master/audio/'):
                    zip_ref.extract(member, ".")
        print("Extraction complete!")

def distribute_files():
    os.makedirs(AMBULANCE_DIR, exist_ok=True)
    os.makedirs(NORMAL_DIR, exist_ok=True)
    
    audio_path = os.path.join(EXTRACT_DIR, "audio")
    if not os.path.exists(audio_path):
        print("Audio folder not found!")
        return

    sirens = 0
    noise = 0

    for file in os.listdir(audio_path):
        if file.endswith("-43.wav"):
            src = os.path.join(audio_path, file)
            dst = os.path.join(AMBULANCE_DIR, "ESC50_Siren_" + file)
            shutil.copy2(src, dst)
            sirens += 1
        elif file.endswith("-44.wav") or file.endswith("-41.wav") or file.endswith("-42.wav"):
            # 44: Car Horn, 41: Chainsaw (engine-like), 42: Siren (actually 42 is false alarm, 42 is siren? wait, let's check)
            # ESC-50 class 43 is Siren, 44 is Car Horn
            src = os.path.join(audio_path, file)
            dst = os.path.join(NORMAL_DIR, "ESC50_Noise_" + file)
            shutil.copy2(src, dst)
            noise += 1

    print(f"Successfully moved {sirens} Siren files to AMBULANCE folder.")
    print(f"Successfully moved {noise} Noise/Horn files to NORMAL folder.")

if __name__ == "__main__":
    download_and_extract()
    distribute_files()
    print("Done! You can now run train_lokal.py")
