import os, glob, numpy as np, librosa
import tensorflow as tf

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
    print("=================================================================")
    print("     AUDIT TOTAL DATASET UTAMA - VERIFIKASI AKURASI KELAS       ")
    print("=================================================================")
    
    # Load model & scaler
    model = tf.keras.models.load_model("siren_classifier_model.h5", compile=False)
    scaler = np.load("siren_scaler.npz")
    mean = scaler['global_mean']
    std = scaler['global_std']

    mismatches = {cat: [] for cat in CATEGORIES}
    counts = {cat: 0 for cat in CATEGORIES}
    matches = {cat: 0 for cat in CATEGORIES}

    print("\nMulai scanning seluruh file audio di dataset utama...")
    
    for cat in CATEGORIES:
        files = glob.glob(f"{cat}/**/*.wav", recursive=True)
        counts[cat] = len(files)
        print(f" -> Memeriksa {cat}: {len(files)} file...")
        
        for f in files:
            try:
                y, _ = librosa.load(f, sr=SAMPLE_RATE)
                mel = extract_melspec(y)
                mel_norm = (mel - mean) / (std + 1e-7)
                mel_input = mel_norm[np.newaxis, ..., np.newaxis]
                
                preds = model.predict(mel_input, verbose=0)[0]
                pred_idx = np.argmax(preds)
                pred_cat = CATEGORIES[pred_idx]
                conf = preds[pred_idx] * 100
                
                if pred_cat == cat:
                    matches[cat] += 1
                else:
                    mismatches[cat].append({
                        "file": f,
                        "pred": pred_cat,
                        "conf": conf
                    })
            except Exception as e:
                print(f"   [ERROR] Gagal proses {f}: {e}")

    print("\n=================================================================")
    print("                    HASIL VERIFIKASI DATASET                     ")
    print("=================================================================")
    total_files = 0
    total_matches = 0
    total_mismatches = 0

    for cat in CATEGORIES:
        tot = counts[cat]
        mat = matches[cat]
        mis = len(mismatches[cat])
        acc = (mat / tot * 100) if tot > 0 else 0
        print(f"Kelas {cat:<10s}: Total={tot:<5d} Match={mat:<5d} Mismatch={mis:<3d} (Akurasi: {acc:.2f}%)")
        total_files += tot
        total_matches += mat
        total_mismatches += mis

    overall_acc = (total_matches / total_files * 100) if total_files > 0 else 0
    print(f"\nTOTAL KESELURUHAN: {total_files} file | MATCH: {total_matches} | MISMATCH: {total_mismatches}")
    print(f"AKURASI DATASET UTAMA: {overall_acc:.2f}%")
    print("=================================================================")

    if total_mismatches > 0:
        print("\nDetail file yang terdeteksi salah/mismatch:")
        for cat in CATEGORIES:
            if mismatches[cat]:
                print(f"\n -> Di folder [{cat}] terdeteksi sebagai:")
                for item in mismatches[cat][:15]: # Tampilkan max 15 per kelas
                    print(f"    * {os.path.basename(item['file'])} -> Terdeteksi: {item['pred']} ({item['conf']:.1f}%)")
                if len(mismatches[cat]) > 15:
                    print(f"    ... dan {len(mismatches[cat]) - 15} file lainnya.")
    else:
        print("\n[SUKSES] 100% data di dataset utama sudah sinkron dengan label kelas masing-masing!")

if __name__ == "__main__":
    main()
