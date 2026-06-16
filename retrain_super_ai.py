import os
import sys
import time
import shutil
import subprocess
import glob
import numpy as np

try:
    import soundfile as sf
    import librosa
except ImportError:
    print("[ERROR] Pastikan librosa dan soundfile sudah diinstall.")
    sys.exit(1)

# Direktori Konfigurasi
TAMBAH_DIR = "TAMBAH_DATASET"
ARCHIVE_DIR = "DATASET_ARCHIVE"
MAIN_DATASET_DIR = "."
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
ESP32_DIR = "sirenmaster_main"

def ensure_dirs():
    for cat in CATEGORIES:
        os.makedirs(os.path.join(TAMBAH_DIR, cat), exist_ok=True)
        os.makedirs(os.path.join(ARCHIVE_DIR, cat), exist_ok=True)
        os.makedirs(os.path.join(MAIN_DATASET_DIR, cat), exist_ok=True)

def mix_with_noise(y, noise_y, snr=0.5):
    if len(noise_y) < len(y):
        noise_y = np.pad(noise_y, (0, len(y) - len(noise_y)), mode='wrap')
    else:
        noise_y = noise_y[:len(y)]
    return y * (1.0 - snr) + noise_y * snr

def process_and_augment_new_data():
    print("\n" + "="*50)
    print(" [1/5] MENYEDOT DATA 'HARD NEGATIVES' BARU")
    print("="*50)
    
    # Ambil beberapa sampel NORMAL (Jalanan) untuk dicampur (Noise Blending)
    normal_files = glob.glob(os.path.join(MAIN_DATASET_DIR, "NORMAL", "*.wav"))
    
    total_processed = 0
    
    for cat in CATEGORIES:
        new_files = glob.glob(os.path.join(TAMBAH_DIR, cat, "*.wav"))
        if not new_files:
            continue
            
        print(f"[*] Ditemukan {len(new_files)} rekaman {cat} baru!")
        
        for fp in new_files:
            filename = os.path.basename(fp)
            base_name = os.path.splitext(filename)[0]
            
            try:
                y, sr = librosa.load(fp, sr=8000)
                
                # Buat variasi ekstrem (Cloning 1 -> 6)
                variations = {
                    "original": y,
                    "pitch_up": librosa.effects.pitch_shift(y, sr=sr, n_steps=2.0),
                    "pitch_down": librosa.effects.pitch_shift(y, sr=sr, n_steps=-2.0),
                    "speed_up": librosa.effects.time_stretch(y, rate=1.2),
                    "speed_down": librosa.effects.time_stretch(y, rate=0.8),
                }
                
                # Noise Blending (hanya jika bukan kelas NORMAL)
                if cat != "NORMAL" and normal_files:
                    try:
                        bg, _ = librosa.load(np.random.choice(normal_files), sr=8000)
                        variations["noise_blend"] = mix_with_noise(y, bg, snr=0.4)
                    except:
                        pass
                
                # Simpan semua variasi langsung ke DATASET Utama
                for v_name, v_audio in variations.items():
                    out_name = f"aug_hardneg_{v_name}_{base_name}.wav"
                    out_path = os.path.join(MAIN_DATASET_DIR, cat, out_name)
                    # Normalisasi volume sebelum disave
                    if np.max(np.abs(v_audio)) > 0:
                        v_audio = v_audio / np.max(np.abs(v_audio))
                    sf.write(out_path, v_audio, sr)
                    
                # Pindahkan file asli ke DATASET_ARCHIVE agar tidak diproses ganda
                shutil.move(fp, os.path.join(ARCHIVE_DIR, cat, filename))
                total_processed += 1
                
            except Exception as e:
                print(f"[!] Gagal memproses {filename}: {e}")
                
    if total_processed == 0:
        print("[*] Tidak ada data baru yang diproses. Lanjut ke Training...")
    else:
        print(f"[*] Berhasil menyuntikkan {total_processed * 6} sampel ekstrem ke dalam otak AI!")

def run_command(cmd, desc):
    print("\n" + "="*50)
    print(f" {desc}")
    print("="*50)
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        print(f"\n[FATAL ERROR] Gagal saat: {desc}")
        sys.exit(1)

def main():
    print(r"""
  _____ _                 _____       _             _       
 / ____| |               |  __ \     | |           (_)      
| (___ | |__  _   _ _ __ | |__) |___ | |__   ___  _ _  ___  
 \___ \| '_ \| | | | '_ \|  _  // _ \| '_ \ / _ \| | |/ __| 
 ____) | | | | |_| | | | | | \ \ (_) | |_) | (_) | | | (__  
|_____/|_| |_|\__,_|_| |_|_|  \_\___/|_.__/ \___/|_|_|\___| 
    AUTO-MLOPS: Belajar Otomatis dari Kesalahan Masa Lalu
    """)
    time.sleep(2)
    
    # 1. Pastikan folder siap
    ensure_dirs()
    
    # 2. Proses data baru (Hard Negatives)
    process_and_augment_new_data()
    
    # 3. Latih Ulang AI (Train)
    # Ini akan memakan waktu lumayan lama
    run_command("python train_lokal.py", "[2/5] MELATIH ULANG OTAK AI (RETRAINING)")
    
    # 4. Export Model ke TFLite dan C++
    run_command("python export_model.py", "[3/5] MENGEKSPOR MODEL KE FORMAT TINYML")
    
    # 5. Salin otomatis ke folder Arduino ESP32
    print("\n" + "="*50)
    print(" [4/5] MEMASANG OTAK BARU KE ARDUINO ESP32")
    print("="*50)
    header_src = "siren_model_data.h"
    header_dst = os.path.join(ESP32_DIR, "siren_model_data.h")
    
    if os.path.exists(header_src):
        shutil.copy2(header_src, header_dst)
        print(f"[*] Sukses menyalin {header_src} ke {ESP32_DIR}/")
    else:
        print(f"[!] Peringatan: File {header_src} tidak ditemukan!")
        
    print("\n" + "="*50)
    print(" [5/5] PROSES MLOPS SELESAI SEMPURNA! [DONE]")
    print("="*50)
    print("AI Anda kini sudah berevolusi menjadi jauh lebih pintar!")
    print("Silakan buka Arduino IDE dan tekan Upload untuk memasukkan otak baru ini ke ESP32.")

if __name__ == "__main__":
    main()
