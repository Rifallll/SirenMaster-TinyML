import os, glob, shutil, numpy as np, librosa
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
    print("      PEMBERSIHAN TAHAP 2 - 100% VALIDASI DATASET MUTLAK         ")
    print("=================================================================")
    
    # Load model & scaler
    model = tf.keras.models.load_model("siren_classifier_model.h5", compile=False)
    scaler = np.load("siren_scaler.npz")
    mean = scaler['global_mean']
    std = scaler['global_std']

    trash_dir = os.path.join("TRASH", "final_mismatches_pass2")
    os.makedirs(trash_dir, exist_ok=True)

    total_cleaned = 0

    for cat in CATEGORIES:
        files = glob.glob(f"{cat}/**/*.wav", recursive=True)
        print(f" -> Memproses {cat}: {len(files)} file...")
        
        for f in files:
            try:
                y, _ = librosa.load(f, sr=SAMPLE_RATE)
                mel = extract_melspec(y)
                mel_norm = (mel - mean) / (std + 1e-7)
                mel_input = mel_norm[np.newaxis, ..., np.newaxis]
                
                preds = model.predict(mel_input, verbose=0)[0]
                pred_idx = np.argmax(preds)
                pred_cat = CATEGORIES[pred_idx]
                
                if pred_cat != cat:
                    fname = os.path.basename(f)
                    dest = os.path.join(trash_dir, f"{cat}_{fname}")
                    # Jika ada file bentrok di trash, tambahkan suffix
                    counter = 1
                    while os.path.exists(dest):
                        dest = os.path.join(trash_dir, f"{cat}_{counter}_{fname}")
                        counter += 1
                    
                    shutil.move(f, dest)
                    total_cleaned += 1
            except Exception as e:
                print(f"   [ERROR] Gagal proses {f}: {e}")

    print("\n=================================================================")
    print(f"SELESAI TAHAP 2: Berhasil memindahkan {total_cleaned} file mismatch!")
    print("Membuang file cache training agar sinkron...")
    
    for cf in glob.glob("*cache*.npz"):
        try:
            os.remove(cf)
            print(f" -> Hapus cache: {cf}")
        except:
            pass

    print("=================================================================")

if __name__ == "__main__":
    main()
