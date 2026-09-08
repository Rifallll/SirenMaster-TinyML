"""
train_v2_fix_police.py
======================
Retrain dengan perbaikan khusus untuk masalah POLICE vs AMBULANCE confusion.

PERBAIKAN UTAMA vs train_lokal.py:
1. Blacklist file POLICE yang ambigu (terdengar seperti ambulans) dari training
2. Oversampling POLICE lebih agresif (underrepresented class)
3. Focal Loss lebih tinggi (gamma=3) untuk fokus pada hard examples
4. Margin loss: memperkuat kesamaan dalam kelas & jarak antar kelas
5. Class weight POLICE lebih tinggi
"""
import os, re, sys, numpy as np, librosa, math, hashlib, glob
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HARD_NEGS_FILES = glob.glob(r"C:\Users\ASUS\Videos\DATASET\NORMAL\hard_neg_*.wav")

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DATASET_PATHS = [r"C:\Users\ASUS\Videos\DATASET"]
OUTPUT_HEADER  = r"C:\Users\ASUS\Videos\DATASET\sirenmaster_main\model.h"
CACHE_PATH     = "siren_v3_pure_audio_cache.npz"

SAMPLE_RATE = 8000
N_FFT       = 256
HOP_LENGTH  = 128
N_MELS      = 40
DURATION    = 4.0
CATEGORIES  = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

# ── Blacklist: file ambigu/rancu dari semua kelas (Ambulance, Firetruck, Normal, Police) ──
ALL_BLACKLIST = set()

# 1. Load dari audit_police_ambiguous_result.txt
BLACKLIST_FILE_1 = r"C:\Users\ASUS\Videos\DATASET\audit_police_ambiguous_result.txt"
if os.path.exists(BLACKLIST_FILE_1):
    with open(BLACKLIST_FILE_1, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if "/" in line:
                ALL_BLACKLIST.add(line.split("/")[-1].strip())
            elif line:
                ALL_BLACKLIST.add(line)

# 2. Load dari audit_total_mismatches.txt (156 file mislabel lintas kelas)
BLACKLIST_FILE_2 = r"C:\Users\ASUS\Videos\DATASET\audit_total_mismatches.txt"
if os.path.exists(BLACKLIST_FILE_2):
    with open(BLACKLIST_FILE_2, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(r'File:\s*([^\s]+)', line)
            if match:
                ALL_BLACKLIST.add(match.group(1).strip())

print(f"[INFO] Loaded {len(ALL_BLACKLIST)} blacklisted ambiguous files across ALL classes (Ambulance, Firetruck, Normal, Police)!")



def get_group_name(file_path):
    base = os.path.splitext(os.path.basename(file_path))[0]
    match = re.match(r'^([a-zA-Z]+_\d+)', base)
    if match:
        return match.group(1)
    cleaned = re.sub(r'\s*\(\d+\)$', '', base)
    cleaned = re.sub(r'_(original|loud|noise_light|shift|seg\d+)$', '', cleaned)
    return cleaned.strip()


def extract_melspec(y, sr=SAMPLE_RATE):
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]

    # Dihapus: Auto-Gain
    # Dihapus: 3-tap LPF
    # Dihapus: DC Removal

    n_frames = (target_len - N_FFT) // HOP_LENGTH + 1
    hamming = np.hamming(N_FFT)
    mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000)

    log_mel_frames = []
    for frame in range(n_frames):
        start = frame * HOP_LENGTH
        fd = y[start:start+N_FFT].copy()

        fft_out = np.fft.rfft(fd * hamming, n=N_FFT)
        power   = np.abs(fft_out) ** 2
        mel_e   = np.dot(mel_fb, power)
        log_mel_frames.append(np.log(mel_e + 1e-9))

    return np.array(log_mel_frames, dtype=np.float32)  # (249, 40)


def process_single_file(fp, label, is_training):
    X_file, y_file = [], []
    target_len = int(SAMPLE_RATE * DURATION)
    fname = os.path.basename(fp)

    # ── Skip blacklisted ambiguous files across ALL classes ──
    if fname in ALL_BLACKLIST:
        return X_file, y_file

    try:
        y, sr = librosa.load(fp, sr=SAMPLE_RATE)

        # Chunk strategy
        chunks = []
        if len(y) >= target_len:
            hop = target_len // 2 if is_training else target_len
            for start in range(0, len(y) - target_len + 1, hop):
                chunks.append(y[start:start+target_len])
        else:
            rms = np.sqrt(np.mean(y**2))
            if rms >= 0.001:
                repeats = int(np.ceil(target_len / len(y)))
                chunks.append(np.tile(y, repeats)[:target_len])

        for chunk in chunks:
            if np.sqrt(np.mean(chunk**2)) < 0.001:
                continue

            feat = extract_melspec(chunk)
            X_file.append(feat)
            y_file.append(label)

            if is_training:
                # Noise augmentation
                noise = np.random.normal(0, 0.003, len(chunk))
                X_file.append(extract_melspec(chunk + noise))
                y_file.append(label)

                # Muffled (distant) simulation
                if label != 2:
                    W = np.random.choice([3, 5, 7])
                    chunk_muffled = np.convolve(chunk, np.ones(W)/W, mode='same')
                    X_file.append(extract_melspec(chunk_muffled))
                    y_file.append(label)

                # Volume scaling
                vol_scale = np.random.uniform(0.1, 1.0)
                X_file.append(extract_melspec(chunk * vol_scale))
                y_file.append(label)

                # Audio mixing with hard negatives
                if label != 2 and len(HARD_NEGS_FILES) > 0 and np.random.rand() < 0.4:
                    try:
                        bg_y, _ = librosa.load(np.random.choice(HARD_NEGS_FILES), sr=SAMPLE_RATE)
                        bg = bg_y[:len(chunk)] if len(bg_y) >= len(chunk) else np.pad(bg_y, (0, len(chunk)-len(bg_y)))
                        rms_s = np.sqrt(np.mean(chunk**2)) + 1e-6
                        rms_b = np.sqrt(np.mean(bg**2)) + 1e-6
                        bg = bg * (rms_s * np.random.uniform(0.5, 1.5) / rms_b)
                        X_file.append(extract_melspec(chunk + bg))
                        y_file.append(label)
                    except:
                        pass

                # ── EXTRA OVERSAMPLING: Khusus kelas POLICE (hanya 1514 file) ──
                # Gandakan 2x dengan pitch shift kecil agar AI belajar lebih banyak variasi polisi
                if label == 3:
                    try:
                        for n_steps in [-1.5, 1.5]:
                            chunk_ps = librosa.effects.pitch_shift(chunk, sr=SAMPLE_RATE, n_steps=n_steps)
                            X_file.append(extract_melspec(chunk_ps))
                            y_file.append(label)
                        # Slow-down simulation (tempo rubato)
                        chunk_slow = librosa.effects.time_stretch(chunk, rate=0.85)
                        chunk_slow = chunk_slow[:target_len] if len(chunk_slow) >= target_len else np.pad(chunk_slow, (0, target_len-len(chunk_slow)))
                        X_file.append(extract_melspec(chunk_slow))
                        y_file.append(label)
                        # Speed-up simulation
                        chunk_fast = librosa.effects.time_stretch(chunk, rate=1.15)
                        chunk_fast = chunk_fast[:target_len] if len(chunk_fast) >= target_len else np.pad(chunk_fast, (0, target_len-len(chunk_fast)))
                        X_file.append(extract_melspec(chunk_fast))
                        y_file.append(label)
                    except:
                        pass

                # Live/guru oversampling
                if ("guru_" in fp.lower() or "live_" in fp.lower()) and label != 2:
                    try:
                        up   = librosa.effects.pitch_shift(chunk, sr=SAMPLE_RATE, n_steps=2.0)
                        down = librosa.effects.pitch_shift(chunk, sr=SAMPLE_RATE, n_steps=-2.0)
                        for _ in range(30):
                            for ch in [feat, extract_melspec(up), extract_melspec(down)]:
                                X_file.append(ch)
                                y_file.append(label)
                    except:
                        pass
    except:
        pass

    return X_file, y_file


def load_split_parallel(file_list, labels, is_training):
    X_all, y_all = [], []
    # Batasi max_workers ke 4 agar CPU laptop tidak terlalu panas & mati sendiri
    max_workers = min(4, os.cpu_count() or 2)
    print(f"    Parallel extraction with {max_workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_single_file, fp, lbl, is_training): i
                   for i, (fp, lbl) in enumerate(zip(file_list, labels))}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 200 == 0 or done == len(file_list):
                print(f"    {done}/{len(file_list)} files processed...", flush=True)
            try:
                X_f, y_f = fut.result()
                X_all.extend(X_f)
                y_all.extend(y_f)
            except:
                pass
    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.int32)


def compute_fingerprint(file_paths):
    sorted_paths = sorted(file_paths)
    s = "".join(f"{fp}:{os.path.getsize(fp)}" for fp in sorted_paths)
    s += "v8_full_clean_retrain"
    return hashlib.md5(s.encode()).hexdigest()


def main():
    print("=" * 60, flush=True)
    print("  SIRENMASTER RETRAIN v2 (POLICE FIX)", flush=True)
    print("=" * 60, flush=True)

    # 1. Scan folders
    print("\n[*] Scanning dataset...", flush=True)
    file_paths, labels, groups = [], [], []
    for idx, cat in enumerate(CATEGORIES):
        cat_files = []
        for base_path in DATASET_PATHS:
            cat_dir = os.path.join(base_path, cat)
            if not os.path.exists(cat_dir):
                continue
            for root, _, files in os.walk(cat_dir):
                for f in files:
                    if f.lower().endswith('.wav'):
                        cat_files.append(os.path.join(root, f))
        print(f"  {cat}: {len(cat_files)} files (before blacklist filter)", flush=True)
        file_paths.extend(cat_files)
        labels.extend([idx] * len(cat_files))
        groups.extend([get_group_name(fp) for fp in cat_files])

    file_paths = np.array(file_paths)
    labels     = np.array(labels)
    groups     = np.array(groups)

    if len(file_paths) == 0:
        raise ValueError("No WAV files found!")

    fingerprint = compute_fingerprint(file_paths)

    # 2. Train/test split
    print("\n[*] Splitting 80/20 by recording group...", flush=True)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(file_paths, labels, groups=groups))
    train_paths,  test_paths  = file_paths[train_idx], file_paths[test_idx]
    train_labels, test_labels = labels[train_idx],     labels[test_idx]
    print(f"  Train: {len(train_paths)}  Test: {len(test_paths)}", flush=True)

    # 3. Cache
    cache_valid = False
    X_train = X_test = y_train = y_test = None
    if os.path.exists(CACHE_PATH):
        try:
            with np.load(CACHE_PATH, allow_pickle=True) as d:
                if 'fingerprint' in d and str(d['fingerprint']) == fingerprint:
                    X_train, y_train = d['X_train'], d['y_train']
                    X_test,  y_test  = d['X_test'],  d['y_test']
                    cache_valid = True
                    print("[INFO] Cache valid, loaded successfully.", flush=True)
                else:
                    print("[INFO] Cache outdated. Rebuilding...", flush=True)
        except Exception as e:
            print(f"[WARN] Cache load failed: {e}. Rebuilding...", flush=True)

    if not cache_valid:
        print("\n[*] Extracting TRAIN features...", flush=True)
        X_train, y_train = load_split_parallel(train_paths, train_labels, True)
        print(f"  Train: {len(X_train)} samples", flush=True)

        print("\n[*] Extracting TEST features...", flush=True)
        X_test, y_test = load_split_parallel(test_paths, test_labels, False)
        print(f"  Test: {len(X_test)} samples", flush=True)

        try:
            np.savez_compressed(CACHE_PATH,
                                X_train=X_train, y_train=y_train,
                                X_test=X_test,   y_test=y_test,
                                fingerprint=fingerprint)
            print("[OK] Cache saved.", flush=True)
        except Exception as e:
            print(f"[WARN] Cache save failed: {e}", flush=True)

    # 4. Scaling
    print("\n[*] Computing Z-score scaling...", flush=True)
    global_mean = np.mean(X_train, axis=(0, 1))
    global_std  = np.std(X_train,  axis=(0, 1))
    global_std[global_std < 1e-6] = 1.0

    np.savez("siren_scaler.npz", global_mean=global_mean, global_std=global_std)
    print("[OK] Saved siren_scaler.npz", flush=True)

    X_train_s = (X_train - global_mean) / global_std
    X_test_s  = (X_test  - global_mean) / global_std
    X_train_s = X_train_s[..., np.newaxis]
    X_test_s  = X_test_s[..., np.newaxis]

    # SpecAugment
    print("\n[*] Applying SpecAugment...", flush=True)
    N, T, F, C = X_train_s.shape
    idx_aug = np.random.choice(N, int(N * 0.3), replace=False)
    for i in idx_aug:
        # Frequency masking
        fm = max(1, int(F * 0.15))
        f0 = np.random.randint(0, F - fm + 1)
        X_train_s[i, :, f0:f0+fm, :] = 0.0
        # Time masking
        tm = max(1, int(T * 0.15))
        t0 = np.random.randint(0, T - tm + 1)
        X_train_s[i, t0:t0+tm, :, :] = 0.0

    perm = np.random.permutation(N)
    X_train_s = X_train_s[perm]
    y_train   = y_train[perm]
    print(f"  Total train samples: {len(X_train_s)}", flush=True)

    # Class distribution
    print("\n[*] Class distribution in train set:", flush=True)
    for i, cat in enumerate(CATEGORIES):
        cnt = (y_train == i).sum()
        print(f"  {cat}: {cnt}", flush=True)

    # 5. Model
    print("\n[*] Importing TensorFlow...", flush=True)
    import tensorflow as tf
    
    # Batasi penggunaan CPU thread TensorFlow agar laptop tidak overheat/restart
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)
    
    from tensorflow.keras import layers, models

    print("\n[*] Building model...", flush=True)
    inp = tf.keras.Input(shape=(X_train_s.shape[1], X_train_s.shape[2], 1))
    x = layers.Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.SeparableConv2D(32, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.SeparableConv2D(48, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(len(CATEGORIES), activation='softmax')(x)
    model = models.Model(inp, out)

    try:
        loss_fn = tf.keras.losses.SparseCategoricalFocalCrossentropy(gamma=3.0)
        print("  Using FocalCrossEntropy (gamma=3.0) — harder examples weighted more", flush=True)
    except AttributeError:
        loss_fn = 'sparse_categorical_crossentropy'
        print("  Focal loss unavailable, using standard CrossEntropy", flush=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=['accuracy']
    )
    model.summary()

    # Class weights — extra weight on POLICE
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = {i: w for i, w in enumerate(cw)}
    cw_dict[3] *= 1.5  # Tambahan bobot 1.5x untuk POLICE
    print(f"\n  Class weights: {cw_dict}", flush=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    ]

    print("\n[*] Training...", flush=True)
    model.fit(
        X_train_s, y_train,
        epochs=30,
        batch_size=512,
        validation_data=(X_test_s, y_test),
        class_weight=cw_dict,
        callbacks=callbacks,
        verbose=1
    )

    model.save("siren_classifier_model.h5")
    print("\n[OK] Keras model saved.", flush=True)

    # 6. Evaluate
    print("\n[*] Evaluating...", flush=True)
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = np.argmax(model.predict(X_test_s), axis=1)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CATEGORIES))

    # 7. Quantize INT8
    print("\n[*] Quantizing to Full INT8...", flush=True)
    rep_data = X_train_s[np.random.choice(len(X_train_s), min(200, len(X_train_s)), replace=False)].astype(np.float32)

    def representative_dataset():
        for i in range(len(rep_data)):
            yield [rep_data[i:i+1]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.inference_input_type  = tf.int8
    conv.inference_output_type = tf.int8
    tflite_model = conv.convert()

    with open('siren_model_quant.tflite', 'wb') as f:
        f.write(tflite_model)
    print(f"  Size: {len(tflite_model)/1024:.2f} KB", flush=True)

    # 8. Export model.h
    print(f"\n[*] Exporting model.h to {OUTPUT_HEADER}...", flush=True)
    hamming = np.hamming(N_FFT).astype(np.float32)
    mel_fb  = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000).T
    n_frames = (int(SAMPLE_RATE * DURATION) - N_FFT) // HOP_LENGTH + 1

    with open(OUTPUT_HEADER, 'w') as f:
        f.write("/*\n * model.h - Siren Classifier (v2 Police Fix)\n */\n\n")
        f.write("#ifndef MODEL_H\n#define MODEL_H\n\n#include <pgmspace.h>\n\n")
        f.write(f"#define SAMPLE_RATE_HZ   {SAMPLE_RATE}\n")
        f.write(f"#define N_FFT_SIZE       {N_FFT}\n")
        f.write(f"#define N_HOP_LENGTH     {HOP_LENGTH}\n")
        f.write(f"#define N_MFCC_COEFF     13\n")
        f.write(f"#define N_MEL_FILTERS    {N_MELS}\n")
        f.write(f"#define N_TIME_FRAMES    {n_frames}\n")
        f.write(f"#define N_FFT_BINS       {N_FFT // 2 + 1}\n")
        f.write(f"#define AUDIO_SAMPLES    {int(SAMPLE_RATE * DURATION)}\n")
        f.write(f"#define NUM_CLASSES      {len(CATEGORIES)}\n\n")

        f.write(f"const float HAMMING_WINDOW[{N_FFT}] PROGMEM = {{\n")
        for i, v in enumerate(hamming):
            if i % 8 == 0: f.write("  ")
            f.write(f"{v:.8f}f")
            if i < N_FFT - 1: f.write(", ")
            if (i+1) % 8 == 0 or i == N_FFT-1: f.write("\n")
        f.write("};\n\n")

        f.write(f"const float MEL_FILTERBANK[{N_MELS}][{N_FFT//2+1}] PROGMEM = {{\n")
        for m in range(N_MELS):
            f.write("  {")
            f.write(", ".join(f"{v:.8f}f" for v in mel_fb[:, m]))
            f.write("}" + (",\n" if m < N_MELS-1 else "\n"))
        f.write("};\n\n")

        f.write(f"const float MEL_MEAN[{N_MELS}] PROGMEM = {{\n  ")
        f.write(", ".join(f"{v:.8f}f" for v in global_mean))
        f.write("\n};\n\n")

        f.write(f"const float MEL_STD[{N_MELS}] PROGMEM = {{\n  ")
        f.write(", ".join(f"{v:.8f}f" for v in global_std))
        f.write("\n};\n\n")

        f.write("#ifdef __has_attribute\n#define MODEL_ALIGN __attribute__((aligned(4)))\n#else\n#define MODEL_ALIGN\n#endif\n\n")
        f.write(f"const unsigned int siren_model_data_len = {len(tflite_model)};\n\n")
        f.write("const unsigned char siren_model_data[] MODEL_ALIGN = {\n")
        for i, val in enumerate(tflite_model):
            if i % 12 == 0: f.write("  ")
            f.write(f"0x{val:02x}")
            if i < len(tflite_model)-1: f.write(", ")
            if (i+1) % 12 == 0 or i == len(tflite_model)-1: f.write("\n")
        f.write("};\n\n#endif // MODEL_H\n")

    print(f"[SUCCESS] model.h exported! Size: {len(tflite_model)/1024:.1f} KB", flush=True)
    print("[DONE] Retrain v2 complete. Upload model.h to ESP32 via Arduino IDE.", flush=True)


if __name__ == "__main__":
    main()
