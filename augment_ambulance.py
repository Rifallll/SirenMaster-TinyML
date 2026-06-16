import os
import glob
import librosa
import soundfile as sf
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings("ignore")

AMBULANCE_DIR = r"c:\Users\ASUS\Videos\DATASET\AMBULANCE"
TARGET_TOTAL = 1605  # Menyesuaikan dengan jumlah POLICE

def add_noise(data, noise_factor=0.015):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def process_single_augmentation(filepath):
    """Melakukan 1 augmentasi acak pada 1 file"""
    basename = os.path.basename(filepath)
    try:
        y, sr = librosa.load(filepath, sr=16000)
        
        # Pilih satu augmentasi secara acak untuk variasi yang merata
        aug_type = random.choice(['pitch', 'stretch', 'noise'])
        
        if aug_type == 'pitch':
            # Pitch Shift (+2 steps atau -2 steps acak)
            steps = random.choice([2, -2])
            y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            out_path = os.path.join(AMBULANCE_DIR, f"aug_pitch_{basename}")
            
        elif aug_type == 'stretch':
            # Time Stretch (sedikit lebih cepat atau lambat)
            rate = random.choice([0.9, 1.1])
            y_aug = librosa.effects.time_stretch(y, rate=rate)
            out_path = os.path.join(AMBULANCE_DIR, f"aug_stretch_{basename}")
            
        else:
            # Background Noise
            y_aug = add_noise(y)
            out_path = os.path.join(AMBULANCE_DIR, f"aug_noise_{basename}")
            
        sf.write(out_path, y_aug, sr)
        return 1
    except Exception as e:
        print(f"Error processing {basename}: {e}")
        return 0

if __name__ == "__main__":
    print("[*] Memulai Data Augmentation cerdas untuk kelas AMBULANCE...")
    
    # Kumpulkan file
    files = glob.glob(os.path.join(AMBULANCE_DIR, "*.wav"))
    original_files = [f for f in files if not os.path.basename(f).startswith("aug_")]
    current_total = len(original_files)
    
    print(f"[*] Ditemukan {current_total} file AMBULANCE asli.")
    
    if current_total >= TARGET_TOTAL:
        print(f"[!] Jumlah file sudah mencapai atau melebihi target ({TARGET_TOTAL}). Tidak perlu augmentasi.")
    elif current_total == 0:
        print("[!] Tidak ada file asli untuk diaugmentasi.")
    else:
        deficit = TARGET_TOTAL - current_total
        print(f"[*] Target total: {TARGET_TOTAL}. Kurang: {deficit} file.")
        
        # Pilih file asli secara acak untuk diaugmentasi sebanyak 'deficit'
        # Jika deficit lebih besar dari jumlah file, kita izinkan duplikasi pilihan (sample with replacement)
        files_to_augment = random.choices(original_files, k=deficit)
        
        print(f"[*] Memproses {deficit} augmentasi (ini mungkin butuh beberapa menit)...")
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(process_single_augmentation, files_to_augment), total=deficit))
            
        total_new = sum(results)
        print(f"\n[+] Berhasil membuat {total_new} file audio baru!")
        
        all_files = glob.glob(os.path.join(AMBULANCE_DIR, "*.wav"))
        print(f"[+] Total data AMBULANCE sekarang: {len(all_files)} files.")
        print("[*] Proses Selesai.")
