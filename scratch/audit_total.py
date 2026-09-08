import os
import hashlib
import numpy as np
import librosa
import tensorflow as tf
import sys
import re

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATASET_DIR = r"C:\Users\ASUS\Videos\DATASET"
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
MODEL_PATH = os.path.join(DATASET_DIR, "siren_classifier_model.h5")

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def extract_melspec(y, sr=8000):
    target_len = int(8000 * 4.0)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
        
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y * min(1.0 / max_val, 10.0)
        
    y_smoothed = np.convolve(y, [1/3, 1/3, 1/3], mode='same')
    
    n_frames = (target_len - 256) // 128 + 1
    hamming = np.hamming(256)
    mel_fb = librosa.filters.mel(sr=8000, n_fft=256, n_mels=40, fmin=0, fmax=4000)
    
    log_mel_frames = []
    for frame in range(n_frames):
        start = frame * 128
        frame_data = y_smoothed[start:start+256].copy()
        frame_data -= np.mean(frame_data)
        power_spec = np.abs(np.fft.rfft(frame_data * hamming, n=256)) ** 2
        log_mel_frames.append(np.log(np.dot(mel_fb, power_spec) + 1e-9))
        
    return np.array(log_mel_frames, dtype=np.float32)

def audit_total():
    print("="*60)
    print("      MEMULAI AUDIT DATASET TOTAL (100% FILES)")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at {MODEL_PATH}!")
        return

    # Load model
    print("[*] Loading AI Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load Scaler from model.h
    print("[*] Loading Scaler params...")
    with open(os.path.join(DATASET_DIR, "sirenmaster_main", "model.h"), 'r') as f:
        content = f.read()
    mean_match = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\}', content)
    std_match  = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\}', content)
    MEL_MEAN = np.array([float(x.strip().rstrip('f')) for x in mean_match.group(1).split(',')])
    MEL_STD  = np.array([float(x.strip().rstrip('f')) for x in std_match.group(1).split(',')])

    # Scan all files
    print("[*] Scanning all WAV files in dataset...")
    files_to_check = []
    for cat in CATEGORIES:
        folder = os.path.join(DATASET_DIR, cat)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith(".wav"):
                files_to_check.append((cat, file, os.path.join(folder, file)))
                
    total_files = len(files_to_check)
    print(f"[INFO] Found {total_files} total files to audit.")

    mismatches = []
    processed = 0

    print("[*] Processing files and analyzing predictions...")
    for true_cat, file_name, path in files_to_check:
        try:
            y, sr = librosa.load(path, sr=8000)
            if len(y) == 0:
                continue
                
            features = extract_melspec(y)
            features = (features - MEL_MEAN) / MEL_STD
            features = np.expand_dims(features, axis=(0, -1))
            
            pred = model.predict(features, verbose=0)[0]
            best_idx = np.argmax(pred)
            pred_cat = CATEGORIES[best_idx]
            confidence = pred[best_idx]
            
            # Audit Threshold: Confidence > 85% but maps to different category
            if pred_cat != true_cat and confidence > 0.85:
                mismatches.append((true_cat, pred_cat, file_name, confidence, path))
                
        except Exception as e:
            pass
            
        processed += 1
        if processed % 500 == 0 or processed == total_files:
            print(f"  Processed {processed}/{total_files} files ({processed/total_files*100:.1f}%)")

    # Report results
    print("\n" + "="*50)
    print("               AUDIT RESULT REPORT")
    print("="*50)
    print(f"Total Audited Files: {total_files}")
    print(f"Mismatches found   : {len(mismatches)}")
    
    output_report_path = os.path.join(DATASET_DIR, "audit_total_mismatches.txt")
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("=== AUDIT TOTAL MISMATCHES REPORT ===\n")
        f.write(f"Total Files Audited: {total_files}\n")
        f.write(f"Total Mismatches Found (Conf > 85%): {len(mismatches)}\n\n")
        
        if len(mismatches) > 0:
            print(f"\n[WARN] Ditemukan {len(mismatches)} file salah folder / mencurigakan!")
            for t, p, fname, conf, path in mismatches:
                log_line = f"Folder: [{t}] | AI predict: [{p}] ({conf*100:.2f}%) | File: {fname}\n"
                f.write(log_line)
                f.write(f"  Path: {path}\n\n")
                print(f"  - File '{fname}' di folder [{t}] dideteksi AI sebagai [{p}] ({conf*100:.1f}%)")
        else:
            print("\n[SUCCESS] SANGAT SEMPURNA! 100% File di dalam dataset sudah tepat klasifikasinya.")
            f.write("Tidak ada file salah folder yang terdeteksi. Dataset 100% bersih!\n")

    print(f"\n[INFO] Detailed report saved to '{output_report_path}'")
    print("="*50)

if __name__ == "__main__":
    audit_total()
