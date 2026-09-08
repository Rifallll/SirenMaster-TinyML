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
    print("================================================================")
    print("   PEMBERSIHAN DATASET TAMBAHAN & ARCHIVE (UNAUDITED FILES)     ")
    print("================================================================")
    
    model = tf.keras.models.load_model("siren_classifier_model.h5", compile=False)
    scaler = np.load("siren_scaler.npz")
    mean = scaler['global_mean']
    std = scaler['global_std']

    trash_dir = os.path.join("TRASH", "unaudited_mismatches")
    os.makedirs(trash_dir, exist_ok=True)

    folders_to_audit = ["TAMBAH_DATASET", "DATASET_ARCHIVE"]
    total_moved = 0
    total_valid_tambah = 0

    for folder_name in folders_to_audit:
        print(f"\n--- Memproses Folder: {folder_name} ---")
        files = glob.glob(f"{folder_name}/**/*.wav", recursive=True)
        if not files:
            continue
            
        for f in sorted(files):
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
                
                fname = os.path.basename(f)
                
                # Jika Mismatch atau conf rendah, pindahkan ke TRASH
                if true_cat != pred_cat or conf < 70.0:
                    dest = os.path.join(trash_dir, f"{true_cat}_{fname}")
                    shutil.move(f, dest)
                    print(f"  [MOVED TO TRASH] {fname} (Folder: {true_cat} | Pred: {pred_cat} {conf:.1f}%)")
                    total_moved += 1
                else:
                    print(f"  [VALID] {fname} (Folder: {true_cat} | Pred: {pred_cat} {conf:.1f}%)")
                    # Jika ini file valid dari TAMBAH_DATASET, salin ke dataset utama agar ikut training berikutnya!
                    if folder_name == "TAMBAH_DATASET":
                        main_dest = os.path.join(true_cat, fname)
                        if not os.path.exists(main_dest):
                            shutil.copy2(f, main_dest)
                            print(f"    -> [ADDED TO MAIN DATASET] Ditambahkan ke folder {true_cat}/")
                            total_valid_tambah += 1
            except Exception as e:
                print(f"  [ERROR] {os.path.basename(f)}: {e}")

    print("\n================================================================")
    print(f"RINGKASAN PEMBERSIHAN:")
    print(f" - File rusak/salah folder dipindahkan ke TRASH : {total_moved} file")
    print(f" - File bersih dari TAMBAH_DATASET ditambahkan  : {total_valid_tambah} file")
    print("================================================================")

    # Hapus cache training karena dataset utama mungkin bertambah
    if total_valid_tambah > 0 or total_moved > 0:
        cache_files = glob.glob("*cache*.npz")
        for cf in cache_files:
            try:
                os.remove(cf)
                print(f"[INFO] Cache '{cf}' dihapus agar training berikutnya sinkron.")
            except:
                pass

if __name__ == "__main__":
    main()
