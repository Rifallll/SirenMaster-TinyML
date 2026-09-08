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

DATASET_DIR = r"c:\Users\ASUS\Videos\DATASET"
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
TARGET_TOTAL = 3849  # Target mengikuti jumlah terbanyak (FIRETRUCK)
MIN_DURATION = 0.5

def add_noise(data, noise_factor=0.015):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def process_single_augmentation(args):
    """Melakukan 1 augmentasi acak pada 1 file"""
    filepath, target_dir = args
    basename = os.path.basename(filepath)
    try:
        y, sr = librosa.load(filepath, sr=16000)
        
        # Pilih satu augmentasi secara acak untuk variasi yang merata
        aug_type = random.choice(['pitch', 'stretch', 'noise'])
        
        if aug_type == 'pitch':
            # Pitch Shift (+2 steps atau -2 steps acak)
            steps = random.choice([2, -2, 1, -1])
            y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            out_path = os.path.join(target_dir, f"aug_pitch_{random.randint(1000,9999)}_{basename}")
            
        elif aug_type == 'stretch':
            # Time Stretch (sedikit lebih cepat atau lambat)
            rate = random.choice([0.8, 0.9, 1.1, 1.2])
            y_aug = librosa.effects.time_stretch(y, rate=rate)
            out_path = os.path.join(target_dir, f"aug_stretch_{random.randint(1000,9999)}_{basename}")
            
        else:
            # Background Noise
            y_aug = add_noise(y)
            out_path = os.path.join(target_dir, f"aug_noise_{random.randint(1000,9999)}_{basename}")
            
        sf.write(out_path, y_aug, sr)
        return 1
    except Exception as e:
        return 0

def clean_short_files():
    print("[*] Tahap 1: Membersihkan file yang terlalu pendek (< 0.5 detik)...")
    deleted_count = 0
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATASET_DIR, cat)
        if not os.path.exists(cat_dir): continue
        
        files = glob.glob(os.path.join(cat_dir, "*.wav"))
        for f in files:
            try:
                duration = librosa.get_duration(path=f)
                if duration < MIN_DURATION:
                    os.remove(f)
                    deleted_count += 1
            except Exception as e:
                pass
    print(f"[+] Selesai membersihkan. Total file dihapus: {deleted_count}")

if __name__ == "__main__":
    print("="*50)
    print("      DATASET BALANCING & AUGMENTATION")
    print("="*50)
    
    clean_short_files()
    
    print("\n[*] Tahap 2: Menyeimbangkan Dataset (Target: ~{} file per kelas)".format(TARGET_TOTAL))
    
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATASET_DIR, cat)
        if not os.path.exists(cat_dir):
            print(f"[!] Direktori {cat_dir} tidak ditemukan.")
            continue
            
        files = glob.glob(os.path.join(cat_dir, "*.wav"))
        current_total = len(files)
        
        print(f"\n[*] Kategori: {cat} | Saat ini: {current_total} file")
        
        if current_total >= TARGET_TOTAL:
            print(f"  [+] Kategori {cat} sudah mencapai atau melebihi target. (Tidak perlu augmentasi)")
            continue
            
        deficit = TARGET_TOTAL - current_total
        print(f"  [-] Kurang {deficit} file. Memulai augmentasi...")
        
        # Hindari file augmentasi sebelumnya sebagai sumber jika memungkinkan
        original_files = [f for f in files if not os.path.basename(f).startswith("aug_")]
        if not original_files:
            original_files = files # Jika semua file sudah hasil augmentasi
            
        if not original_files:
            print(f"  [!] Tidak ada file di kategori {cat} untuk diaugmentasi.")
            continue
            
        # Pilih file sumber secara acak
        files_to_augment = random.choices(original_files, k=deficit)
        args_list = [(f, cat_dir) for f in files_to_augment]
        
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(process_single_augmentation, args_list), total=deficit, desc=f"Augmenting {cat}"))
            
        total_new = sum(results)
        final_total = len(glob.glob(os.path.join(cat_dir, "*.wav")))
        print(f"  [+] Augmentasi {cat} selesai. Dibuat: {total_new} | Total sekarang: {final_total}")
        
    print("\n[SUCCESS] Proses penyeimbangan dataset selesai!")
