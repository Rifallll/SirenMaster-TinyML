import os, glob, numpy as np, librosa
import tensorflow as tf

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
SAMPLE_RATE = 8000
DURATION = 4.0
N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40

def extract_melspec(y, sr=SAMPLE_RATE):
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]

    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        gain = min(1.0 / max_val, 10.0)
        y = y * gain

    y_smoothed = np.convolve(y, [1/3, 1/3, 1/3], mode='same')
    n_frames = (target_len - N_FFT) // HOP_LENGTH + 1
    hamming = np.hamming(N_FFT)
    mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000)

    log_mel_frames = []
    for frame in range(n_frames):
        start = frame * HOP_LENGTH
        fd = y_smoothed[start:start+N_FFT].copy()
        fd -= np.mean(fd)
        fft_out = np.fft.rfft(fd * hamming, n=N_FFT)
        power = np.abs(fft_out) ** 2
        mel_e = np.dot(mel_fb, power)
        log_mel_frames.append(np.log(mel_e + 1e-9))

    return np.array(log_mel_frames, dtype=np.float32)

def main():
    print("====================================================")
    print("   AUDIT DATASET YANG BELUM TERMASUK AUDIT TOTAL    ")
    print("====================================================")
    
    model_path = "siren_classifier_model.h5"
    scaler_path = "siren_scaler.npz"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("[ERROR] Model atau scaler tidak ditemukan!")
        return

    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = np.load(scaler_path)
    mean = scaler['global_mean']
    std = scaler['global_std']

    folders_to_audit = ["TAMBAH_DATASET", "DATASET_ARCHIVE"]
    
    for folder_name in folders_to_audit:
        print(f"\n--- Memeriksa Folder: {folder_name} ---")
        files = glob.glob(f"{folder_name}/**/*.wav", recursive=True)
        if not files:
            print(f"  [INFO] Tidak ada file .wav di {folder_name}")
            continue
            
        correct = 0
        mismatches = 0
        
        for f in sorted(files):
            # Tentukan true label dari path folder
            upper_path = f.upper()
            true_cat = "UNKNOWN"
            for cat in CATEGORIES:
                if f"/{cat}/" in upper_path.replace("\\", "/") or f"_{cat}" in upper_path:
                    true_cat = cat
                    break
            if true_cat == "UNKNOWN" and "POLICE_KARANTINA" in upper_path:
                true_cat = "POLICE"
                
            try:
                y, _ = librosa.load(f, sr=SAMPLE_RATE)
                mel = extract_melspec(y)
                mel_norm = (mel - mean) / (std + 1e-7)
                mel_input = mel_norm[np.newaxis, ..., np.newaxis]
                
                preds = model.predict(mel_input, verbose=0)[0]
                pred_idx = np.argmax(preds)
                pred_cat = CATEGORIES[pred_idx]
                conf = preds[pred_idx] * 100
                
                is_match = (true_cat == pred_cat)
                status = "MATCH" if is_match else "MISMATCH"
                if is_match:
                    correct += 1
                else:
                    mismatches += 1
                    
                print(f"  [{status}] {os.path.basename(f)} | Folder: {true_cat} -> AI Pred: {pred_cat} ({conf:.1f}%)")
            except Exception as e:
                print(f"  [ERROR] {os.path.basename(f)}: {e}")
                
        print(f"Summary untuk {folder_name}: Total={len(files)}, Cocok={correct}, Mismatch={mismatches}")

if __name__ == "__main__":
    main()
