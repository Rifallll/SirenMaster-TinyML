import os
import hashlib
import numpy as np
import librosa
from collections import defaultdict
import tensorflow as tf
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATASET_DIR = "C:\\Users\\ASUS\\Videos\\DATASET"
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

def audit_dataset():
    print("="*50)
    print("MAMULAI AUDIT DATASET TOTAL...")
    print("="*50)
    
    hashes = defaultdict(list)
    corrupted = []
    files_to_check = []
    
    print("[1] Memindai file duplikat & rusak...")
    for cat in CATEGORIES:
        folder = os.path.join(DATASET_DIR, cat)
        if not os.path.exists(folder): continue
        for file in os.listdir(folder):
            if not file.endswith(".wav"): continue
            path = os.path.join(folder, file)
            
            # Check Hash for duplicates
            file_hash = get_file_hash(path)
            hashes[file_hash].append((cat, file))
            
            # Check corruption & save for AI test
            try:
                y, sr = librosa.load(path, sr=8000)
                if len(y) == 0:
                    corrupted.append(path)
                else:
                    files_to_check.append((cat, file, path, y))
            except Exception as e:
                corrupted.append(path)

    # 1. Report Duplicates
    print("\n[!] HASIL AUDIT DUPLIKAT LINTAS KELAS:")
    cross_class_dupes = 0
    for file_hash, file_list in hashes.items():
        if len(file_list) > 1:
            cats = set([x[0] for x in file_list])
            if len(cats) > 1: # Found identical files in DIFFERENT folders!
                print(f"    BAHAYA: File kembar ditemukan di folder berbeda: {file_list}")
                cross_class_dupes += 1
    if cross_class_dupes == 0:
        print("    Aman! Tidak ada file suara yang nyasar ke folder kelas lain.")

    # 2. Report Corrupted
    print(f"\n[!] HASIL AUDIT KERUSAKAN FILE: Ditemukan {len(corrupted)} file rusak.")

    # 3. Label Error Detection using AI
    print("\n[2] Menjalankan AI untuk mendeteksi file yang salah kamar (Label Error)...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        
        import re
        with open(os.path.join(DATASET_DIR, "sirenmaster_main", "model.h"), 'r') as f:
            content = f.read()
        mean_match = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\}', content)
        std_match  = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\}', content)
        MEL_MEAN = np.array([float(x.strip().rstrip('f')) for x in mean_match.group(1).split(',')])
        MEL_STD  = np.array([float(x.strip().rstrip('f')) for x in std_match.group(1).split(',')])
        
    except Exception as e:
        print("Gagal memuat model. Lewati tes AI.")
        return

    suspicious_files = []
    
    # We will sample 100 files per category to speed up audit
    import random
    random.shuffle(files_to_check)
    sampled = files_to_check[:400] # Check 400 random files
    
    for true_cat, file_name, path, y in sampled:
        features = extract_melspec(y)
        features = (features - MEL_MEAN) / MEL_STD
        features = np.expand_dims(features, axis=(0, -1))
        
        pred = model.predict(features, verbose=0)[0]
        best_idx = np.argmax(pred)
        pred_cat = CATEGORIES[best_idx]
        confidence = pred[best_idx]
        
        # Jika AI SANGAT YAKIN ( > 95%) tapi tebakannya BEDA dengan nama folder aslinya
        if pred_cat != true_cat and confidence > 0.95:
            suspicious_files.append((true_cat, pred_cat, file_name, confidence))

    print("\n[!] HASIL AUDIT SALAH KAMAR (Kecurigaan AI):")
    if len(suspicious_files) == 0:
        print("    Sangat Aman! Dari sampel yang dites, AI setuju dengan semua peletakan folder Anda.")
    else:
        print(f"    Ditemukan {len(suspicious_files)} file yang kemungkinan SALAH FOLDER (Misalnya Polisi masuk ke Damkar):")
        for t, p, f, c in suspicious_files:
            print(f"    - File '{f}' ada di folder [{t}], tapi AI sangat yakin ({c*100:.1f}%) itu suara [{p}]!")

if __name__ == "__main__":
    audit_dataset()
