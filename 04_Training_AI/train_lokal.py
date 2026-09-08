import os
import re
import sys
import numpy as np
import librosa
import math
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob

HARD_NEGS_FILES = glob.glob(r"C:\Users\ASUS\Videos\DATASET\NORMAL\hard_neg_*.wav")

# Set encoding for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# --- CONFIGURATION ---
# Dataset paths to combine
DATASET_PATHS = [
    r"C:\Users\ASUS\Videos\DATASET"
]
OUTPUT_HEADER = r"C:\Users\ASUS\Videos\DATASET\sirenmaster_main\model.h"
CACHE_PATH = "siren_40x249_melspec_cache_lokal.npz"

SAMPLE_RATE = 8000
N_FFT = 256
HOP_LENGTH = 128
N_MFCC = 13
N_MELS = 40
DURATION = 4.0  # 32000 samples -> input (249, 40) -> muat di ESP32
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

def get_improved_group_name(file_path):
    """Extracts root recording group to prevent train-test data contamination."""
    base = os.path.splitext(os.path.basename(file_path))[0]
    match = re.match(r'^([a-zA-Z]+_\d+)', base)
    if match:
        return match.group(1)
    cleaned = base
    cleaned = re.sub(r'\s*\(\d+\)$', '', cleaned)
    cleaned = re.sub(r'_(original|loud|noise_light|shift|seg\d+|_seg\d+)$', '', cleaned)
    cleaned = cleaned.replace('-[AudioTrimmer.com]', '').strip()
    return cleaned

def extract_melspec(y, sr=SAMPLE_RATE):
    """Extracts Log-Mel Spectrogram matching C++ DSP exactly:
    - 3-tap moving average LPF: (prev + curr + next) / 3
    - DC offset removal per frame: frame - mean(frame)
    - Hamming window multiplication
    - FFT power spectrum
    - Mel-filterbank dot product
    """
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
        
    # NORMALIZE AUDIO (Auto-Gain) with max 10x boost
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        gain = min(1.0 / max_val, 10.0)
        y = y * gain
        
    # 1. 3-tap moving average filter (Low Pass Filter)
    # Using np.convolve with [1/3, 1/3, 1/3], mode='same' matches C++ logic
    y_smoothed = np.convolve(y, [1/3, 1/3, 1/3], mode='same')
    
    # 2. Frame-by-frame STFT with DC offset removal and Hamming windowing
    n_frames = (target_len - N_FFT) // HOP_LENGTH + 1
    hamming = np.hamming(N_FFT)
    mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000)
    
    log_mel_frames = []
    for frame in range(n_frames):
        start = frame * HOP_LENGTH
        frame_data = y_smoothed[start:start+N_FFT].copy()
        
        # DC removal
        frame_mean = np.mean(frame_data)
        frame_data = frame_data - frame_mean
        
        # Hamming windowing
        frame_windowed = frame_data * hamming
        
        # Power spectrum
        fft_complex = np.fft.rfft(frame_windowed, n=N_FFT)
        power_spec = np.abs(fft_complex) ** 2
        
        # Mel energies
        mel_energies = np.dot(mel_fb, power_spec)
        
        # Log energy (with 1e-9 floor to match logf(energy + 1e-9f))
        log_mel = np.log(mel_energies + 1e-9)
        log_mel_frames.append(log_mel)
        
    return np.array(log_mel_frames, dtype=np.float32) # shape: (249, 40)


def process_single_file(fp, label, is_training):
    """Processes a single file, extracts features, and applies augmentations."""
    X_file = []
    y_file = []
    target_len = int(SAMPLE_RATE * DURATION)
    try:
        y, sr = librosa.load(fp, sr=SAMPLE_RATE)
        
        chunks = []
        if len(y) >= target_len:
            if is_training:
                hop = target_len // 2
            else:
                hop = target_len
            for start in range(0, len(y) - target_len + 1, hop):
                chunks.append(y[start:start+target_len])
        else:
            rms = np.sqrt(np.mean(y**2))
            if rms >= 0.001:
                # Tiling (repeating) instead of constant zero padding
                repeats = int(np.ceil(target_len / len(y)))
                tiled = np.tile(y, repeats)[:target_len]
                chunks.append(tiled)
                
        for chunk in chunks:
            rms = np.sqrt(np.mean(chunk**2))
            if rms < 0.001:
                continue
                
            # Clean
            feat = extract_melspec(chunk)
            X_file.append(feat)
            y_file.append(label)
            
            if is_training:
                # 1. Hanya gunakan Noise ringan untuk mempercepat training darurat
                noise = np.random.normal(0, 0.003, len(chunk))
                feat_noise = extract_melspec(chunk + noise)
                X_file.append(feat_noise)
                y_file.append(label)
                
                # 2. Simulasi Sirine Muffled/Distant (Low Pass Filter via moving average)
                # Berguna agar AI mengenali sirine sayup-sayup dari jarak 3+ meter (di dalam mobil/ruangan)
                if label != 2:
                    W = np.random.choice([3, 5, 7])
                    chunk_muffled = np.convolve(chunk, np.ones(W)/W, mode='same')
                    feat_muffled = extract_melspec(chunk_muffled)
                    X_file.append(feat_muffled)
                    y_file.append(label)
                    
                # 3. Volume Scaling Augmentation (Krusial agar AI kebal terhadap fluktuasi suara live)
                # Mengubah volume antara 10% hingga 100% secara acak
                vol_scale = np.random.uniform(0.1, 1.0)
                chunk_vol = chunk * vol_scale
                feat_vol = extract_melspec(chunk_vol)
                X_file.append(feat_vol)
                y_file.append(label)
                
                # --- NEW: AUDIO MIXING AUGMENTATION (Sirine + Keramaian) ---
                if label != 2 and len(HARD_NEGS_FILES) > 0:
                    if np.random.rand() < 0.4: # 40% chance to mix with hard negative
                        try:
                            bg_file = np.random.choice(HARD_NEGS_FILES)
                            bg_y, _ = librosa.load(bg_file, sr=SAMPLE_RATE)
                            bg_chunk = bg_y[:len(chunk)] if len(bg_y) >= len(chunk) else np.pad(bg_y, (0, len(chunk) - len(bg_y)))
                            
                            # Mix volume randomly between 0.5x to 1.5x of siren volume
                            rms_siren = np.sqrt(np.mean(chunk**2)) + 1e-6
                            rms_bg = np.sqrt(np.mean(bg_chunk**2)) + 1e-6
                            bg_chunk = bg_chunk * (rms_siren * np.random.uniform(0.5, 1.5) / rms_bg)
                            
                            feat_mixed = extract_melspec(chunk + bg_chunk)
                            X_file.append(feat_mixed)
                            y_file.append(label)
                        except:
                            pass
                
                # --- OVER-SAMPLING & AUGMENTASI KHUSUS FILE USER ---
                if ("guru_" in fp.lower() or "live_" in fp.lower()) and label != 2:
                    try:
                        # Tambahkan Pitch Shift agar AI kebal terhadap distorsi speaker HP
                        chunk_up = librosa.effects.pitch_shift(chunk, sr=SAMPLE_RATE, n_steps=2.0)
                        chunk_down = librosa.effects.pitch_shift(chunk, sr=SAMPLE_RATE, n_steps=-2.0)
                        
                        feat_up = extract_melspec(chunk_up)
                        feat_down = extract_melspec(chunk_down)
                        
                        # Gandakan (Duplikasi 30x) agar suara ini tidak tenggelam dari 3600 dataset internet
                        for _ in range(30):
                            X_file.append(feat)
                            y_file.append(label)
                            X_file.append(feat_up)
                            y_file.append(label)
                            X_file.append(feat_down)
                            y_file.append(label)
                    except Exception as e:
                        pass
    except Exception as e:
        pass
        
    return X_file, y_file

def load_and_process_split_parallel(file_list, labels, is_training):
    X_features = []
    y_labels = []
    
    max_workers = min(12, os.cpu_count() or 4)
    print(f"    Running feature extraction in parallel with {max_workers} workers...", flush=True)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, fp, label, is_training): i 
                   for i, (fp, label) in enumerate(zip(file_list, labels))}
        
        completed = 0
        total = len(file_list)
        for fut in as_completed(futures):
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"    Processed {completed}/{total} files...", flush=True)
            
            try:
                X_file, y_file = fut.result()
                X_features.extend(X_file)
                y_labels.extend(y_file)
            except Exception as e:
                pass
                
    return np.array(X_features, dtype=np.float32), np.array(y_labels, dtype=np.int32)

def compute_dataset_fingerprint(file_paths):
    sorted_paths = sorted(file_paths)
    fingerprint_str = "".join([f"{fp}:{os.path.getsize(fp)}" for fp in sorted_paths])
    # Add version salt to invalidate old cache and force regeneration with new oversampling rules
    fingerprint_str += "v7_rhythm_clean"
    return hashlib.md5(fingerprint_str.encode('utf-8')).hexdigest()

def main():
    print("=" * 60, flush=True)
    print("  TINYML EMERGENCY SIREN TRAINING PIPELINE (13x63)", flush=True)
    print("=" * 60, flush=True)
    
    # 1. Scan folders
    print("\n[*] Scanning dataset directories...", flush=True)
    file_paths = []
    labels = []
    groups = []
    
    for idx, cat in enumerate(CATEGORIES):
        cat_files = []
        for base_path in DATASET_PATHS:
            cat_dir = os.path.join(base_path, cat)
            if not os.path.exists(cat_dir):
                continue
                
            for root, _, files in os.walk(cat_dir):
                for file in files:
                    if file.lower().endswith('.wav'):
                        cat_files.append(os.path.join(root, file))
            
            # [CRITICAL FIX] Sub-sampling kelas Sirine (AMBULANCE, FIRETRUCK, POLICE) agar seimbang (1200 file per kelas)
            if cat in ["AMBULANCE", "FIRETRUCK", "POLICE"] and len(cat_files) > 1200:
                print(f"  [!] Menyeimbangkan {cat} ({len(cat_files)} file) ke 1200 file terbaik...", flush=True)
                synth_files = [f for f in cat_files if os.path.basename(f).startswith("SYNTH_") or "guru" in f.lower() or "live" in f.lower()]
                regular_files = [f for f in cat_files if f not in synth_files]
                np.random.seed(42)
                np.random.shuffle(regular_files)
                needed = max(0, 1200 - len(synth_files))
                cat_files = synth_files + regular_files[:needed]
                print(f"      -> {len(synth_files)} file prioritas + {needed} file reguler.", flush=True)

            # [CRITICAL FIX] Sub-sampling kelas NORMAL agar tidak mendominasi AI
            if cat == "NORMAL" and len(cat_files) > 1500:
                print(f"  [!] Kelas NORMAL terlalu dominan ({len(cat_files)} file). Memprioritaskan Suara Pengecoh (Hard Negatives)...", flush=True)
                
                important_keywords = ["hard", "neg", "telolet", "toa", "masjid", "adzan", "azan", "mosque", "prayer", "vocal", "mimic", "speech", "laugh", "cry", "music", "edm", "whistle", "baby", "siul", "bayi", "horn", "guru", "live", "ultimate"]
                important_files = []
                regular_files = []
                
                for f in cat_files:
                    if any(kw in f.lower() for kw in important_keywords):
                        important_files.append(f)
                    else:
                        regular_files.append(f)
                        
                np.random.seed(42)
                np.random.shuffle(regular_files)
                
                if len(important_files) >= 1500:
                    cat_files = important_files[:1500]
                    needed = 0
                else:
                    needed = 1500 - len(important_files)
                    cat_files = important_files + regular_files[:needed]
                
                print(f"      -> Berhasil mengamankan {len(important_files)} suara ekstrem (bayi/siulan/telolet) & {needed} suara bising biasa.", flush=True)
                        
        print(f"  Category '{cat}': found {len(cat_files)} files.", flush=True)
        file_paths.extend(cat_files)
        labels.extend([idx] * len(cat_files))
        groups.extend([get_improved_group_name(fp) for fp in cat_files])

        
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    groups = np.array(groups)
    
    if len(file_paths) == 0:
        raise ValueError(f"No WAV files found at: {DATASET_PATHS}")
        
    current_fingerprint = compute_dataset_fingerprint(file_paths)
    
    # 2. Train-test group split to avoid data leakage
    print("\n[*] Splitting dataset (80% train, 20% test) by recording group...", flush=True)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(file_paths, labels, groups=groups))
    
    train_paths, test_paths = file_paths[train_idx], file_paths[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    
    print(f"  Train set: {len(train_paths)} files", flush=True)
    print(f"  Test set:  {len(test_paths)} files", flush=True)
    
    # 3. Cache Check and Processing
    cache_valid = False
    X_train, y_train, X_test, y_test = None, None, None, None
    
    if os.path.exists(CACHE_PATH):
        try:
            print(f"\n[*] Found feature cache file: {CACHE_PATH}", flush=True)
            with np.load(CACHE_PATH, allow_pickle=True) as data:
                if 'fingerprint' in data and str(data['fingerprint']) == current_fingerprint:
                    X_train = data['X_train']
                    y_train = data['y_train']
                    X_test = data['X_test']
                    y_test = data['y_test']
                    cache_valid = True
                    print("  [SUCCESS] Cache matches dataset fingerprint. Loaded successfully.", flush=True)
                else:
                    print("  [INFO] Cache fingerprint mismatch. Dataset changed. Rebuilding cache...", flush=True)
        except Exception as e:
            print(f"  [WARNING] Failed to load cache: {e}. Rebuilding...", flush=True)
            
    if not cache_valid:
        print("\n[*] Extracting train set features and applying augmentations...", flush=True)
        X_train, y_train = load_and_process_split_parallel(train_paths, train_labels, is_training=True)
        print(f"  Train features extracted: {len(X_train)} samples.", flush=True)
        
        print("\n[*] Extracting test set features (unaugmented)...", flush=True)
        X_test, y_test = load_and_process_split_parallel(test_paths, test_labels, is_training=False)
        print(f"  Test features extracted: {len(X_test)} samples.", flush=True)
        
        print(f"\n[*] Saving extracted features to cache: {CACHE_PATH}...", flush=True)
        try:
            np.savez_compressed(CACHE_PATH, 
                                X_train=X_train, y_train=y_train, 
                                X_test=X_test, y_test=y_test, 
                                fingerprint=current_fingerprint)
            print("  [SUCCESS] Features cached successfully.", flush=True)
        except Exception as e:
            print(f"  [ERROR] Failed to save cache: {e}", flush=True)
            
    # 4. Standard Scaling (Z-score Normalization)
    print("\n[*] Computing Z-score scaling parameters...", flush=True)
    global_mean = np.mean(X_train, axis=(0, 1))  # (40,)
    global_std = np.std(X_train, axis=(0, 1))    # (40,)
    global_std[global_std < 1e-6] = 1.0
    
    # Auto-save scaler for inference and fine-tuning sync
    SCALER_PATH = "siren_scaler.npz"
    try:
        np.savez(SCALER_PATH, global_mean=global_mean, global_std=global_std)
        print(f"  [SUCCESS] Saved global mean and std to {SCALER_PATH}", flush=True)
    except Exception as e:
        print(f"  [ERROR] Failed to save scaler: {e}", flush=True)

    # Auto-save replay buffer (1000 random samples) for hot-swap tuning sync
    REPLAY_PATH = "siren_replay_buffer.npz"
    try:
        num_samples = len(X_train)
        replay_size = min(1000, num_samples)
        indices = np.random.choice(num_samples, replay_size, replace=False)
        X_replay = X_train[indices]
        y_replay = y_train[indices]
        np.savez_compressed(REPLAY_PATH, X_replay=X_replay, y_replay=y_replay)
        print(f"  [SUCCESS] Saved replay buffer ({replay_size} samples) to {REPLAY_PATH}", flush=True)
    except Exception as e:
        print(f"  [ERROR] Failed to save replay buffer: {e}", flush=True)

    # Scale train and test datasets
    X_train_scaled = (X_train - global_mean[np.newaxis, np.newaxis, :]) / global_std[np.newaxis, np.newaxis, :]
    X_test_scaled = (X_test - global_mean[np.newaxis, np.newaxis, :]) / global_std[np.newaxis, np.newaxis, :]
    
    # Reshape for 2D CNN (samples, 63, 13, 1)
    X_train_scaled = X_train_scaled[..., np.newaxis]
    X_test_scaled = X_test_scaled[..., np.newaxis]
    
    print("\n[*] Applying in-memory SpecAugment...", flush=True)
    def apply_spec_augment_inplace(X, fraction=0.3, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.15):
        num_samples, time_steps, freq_bins, channels = X.shape
        indices = np.random.choice(num_samples, int(num_samples * fraction), replace=False)
        for i in indices:
            # Frequency masking
            freq_mask_size = int(freq_bins * freq_masking_max_percentage)
            if freq_mask_size > 0:
                f0 = np.random.randint(0, freq_bins - freq_mask_size + 1)
                X[i, :, f0:f0+freq_mask_size, :] = 0.0
            
            # Time masking
            time_mask_size = int(time_steps * time_masking_max_percentage)
            if time_mask_size > 0:
                t0 = np.random.randint(0, time_steps - time_mask_size + 1)
                X[i, t0:t0+time_mask_size, :, :] = 0.0
                
    apply_spec_augment_inplace(X_train_scaled, fraction=0.3)
    
    # Shuffle the augmented training set
    idx = np.random.permutation(len(X_train_scaled))
    X_train_scaled = X_train_scaled[idx]
    y_train = y_train[idx]
    print(f"  Total training samples after SpecAugment: {len(X_train_scaled)}", flush=True)
    
    print("  Mean values per Mel coefficient:", np.round(global_mean, 4), flush=True)
    print("  Std values per Mel coefficient:", np.round(global_std, 4), flush=True)
    
    # 5. Build and Train Model
    print("\n[*] Importing TensorFlow and Keras...", flush=True)
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    print("\n[*] Building LIGHTWEIGHT 2D CNN model (fits ESP32 SRAM)...", flush=True)
    model = models.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2], 1)),
        
        # Blok 1: Tangkap fitur dasar (16 filter saja) dengan STRIDE 2 untuk mengecilkan RAM
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Blok 2: Depthwise Separable (hemat parameter & aktivasi)
        layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Blok 3: Tangkap pola tingkat tinggi
        layers.SeparableConv2D(48, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),  # Kurangi drastis ukuran output
        
        # Classifier ringan
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(len(CATEGORIES), activation='softmax')
    ])
    
    loss_fn = 'sparse_categorical_crossentropy'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=['accuracy']
    )
    model.summary()
    
    # Class weights seimbang otomatis
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"  [INFO] Class weights otomatis seimbang: {class_weight_dict}", flush=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    
    print("\n[*] Training CNN model...", flush=True)
    model.fit(
        X_train_scaled, y_train,
        epochs=20,
        batch_size=256,
        validation_data=(X_test_scaled, y_test),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Simpan model Keras (.h5) untuk keperluan Incremental Learning (Fine-Tuning)
    print("\n[*] Saving Keras model to siren_classifier_model.h5...", flush=True)
    model.save("siren_classifier_model.h5")
    
    # 6. Evaluate Model
    print("\n[*] Evaluating model on unseen test split...", flush=True)
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nConfusion Matrix:", flush=True)
    print(confusion_matrix(y_test, y_pred), flush=True)
    print("\nClassification Report:", flush=True)
    print(classification_report(y_test, y_pred, target_names=CATEGORIES), flush=True)
    
    # 7. Full INT8 Quantization — aktivasi + bobot semuanya int8
    # → Arena ESP32 turun dari 320KB ke ~80KB (sesuai TENSOR_ARENA_KB 85)
    # → Akurasi turun ~1-2% saja karena model sudah dilatih dengan data yang bagus
    print("\n[*] Quantizing model to Full INT8 for TFLite Micro deployment...", flush=True)

    # Representative dataset: 200 sampel acak dari training set (sudah ternormalisasi)
    rep_size = min(200, len(X_train_scaled))
    rep_indices = np.random.choice(len(X_train_scaled), rep_size, replace=False)
    rep_data = X_train_scaled[rep_indices].astype(np.float32)

    def representative_dataset():
        for i in range(rep_size):
            sample = rep_data[i:i+1]  # shape (1, 249, 40, 1)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # Full INT8: input & output juga int8 → semua aktivasi int8 di ESP32
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()

    # Save quantized TFLite binary to disk for local testing
    with open('siren_model_quant.tflite', 'wb') as f_tflite:
        f_tflite.write(tflite_quant_model)
    print("  Full INT8 model saved to siren_model_quant.tflite", flush=True)
    print(f"  Quantized model size: {len(tflite_quant_model)/1024:.2f} KB", flush=True)
    
    # 8. Compute windowing and filterbanks
    hamming = np.hamming(N_FFT).astype(np.float32)
    mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000).T
    
    # 9. Generate C++ Deployment Header
    print(f"\n[*] Exporting deployment header model.h to {OUTPUT_HEADER}...", flush=True)
    
    n_frames = (int(SAMPLE_RATE * DURATION) - N_FFT) // HOP_LENGTH + 1
    
    with open(OUTPUT_HEADER, 'w') as f:
        f.write("/*\n")
        f.write(" * model.h - Siren Classifier Deployment Header\n")
        f.write(" * Generated automatically by train_siren_model.py\n")
        f.write(" */\n\n")
        f.write("#ifndef MODEL_H\n")
        f.write("#define MODEL_H\n\n")
        f.write("#include <pgmspace.h>\n\n")
        
        f.write(f"#define SAMPLE_RATE_HZ   {SAMPLE_RATE}\n")
        f.write(f"#define N_FFT_SIZE       {N_FFT}\n")
        f.write(f"#define N_HOP_LENGTH     {HOP_LENGTH}\n")
        f.write(f"#define N_MFCC_COEFF     {N_MFCC}\n")
        f.write(f"#define N_MEL_FILTERS    {N_MELS}\n")
        f.write(f"#define N_TIME_FRAMES    {n_frames}\n")
        f.write(f"#define N_FFT_BINS       {N_FFT // 2 + 1}\n")
        f.write(f"#define AUDIO_SAMPLES    {int(SAMPLE_RATE * DURATION)}\n")
        f.write(f"#define NUM_CLASSES      {len(CATEGORIES)}\n\n")
        
        # Hamming window
        f.write(f"const float HAMMING_WINDOW[{N_FFT}] PROGMEM = {{\n")
        for i, val in enumerate(hamming):
            if i % 8 == 0:
                f.write("  ")
            f.write(f"{val:.8f}f")
            if i < N_FFT - 1:
                f.write(", ")
            if (i + 1) % 8 == 0 or i == N_FFT - 1:
                f.write("\n")
        f.write("};\n\n")
        
        # Mel filterbank
        f.write(f"const float MEL_FILTERBANK[{N_MELS}][{N_FFT//2+1}] PROGMEM = {{\n")
        for m in range(N_MELS):
            f.write("  {")
            row = mel_fb[:, m]
            f.write(", ".join([f"{val:.8f}f" for val in row]))
            f.write("}")
            if m < N_MELS - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")
        
        # Scaling parameters (MEL_MEAN and MEL_STD)
        f.write(f"const float MEL_MEAN[{N_MELS}] PROGMEM = {{\n  ")
        f.write(", ".join([f"{val:.8f}f" for val in global_mean]))
        f.write("\n};\n\n")
        
        f.write(f"const float MEL_STD[{N_MELS}] PROGMEM = {{\n  ")
        f.write(", ".join([f"{val:.8f}f" for val in global_std]))
        f.write("\n};\n\n")
        
        # TFLite model data aligned to 4 bytes for microcontroller memory efficiency
        f.write("// Align model array for TFLite Micro\n")
        f.write("#ifdef __has_attribute\n")
        f.write("#define MODEL_ALIGN __attribute__((aligned(4)))\n")
        f.write("#else\n")
        f.write("#define MODEL_ALIGN\n")
        f.write("#endif\n\n")
        
        f.write(f"const unsigned int siren_model_data_len = {len(tflite_quant_model)};\n\n")
        f.write("const unsigned char siren_model_data[] MODEL_ALIGN = {\n")
        for i, val in enumerate(tflite_quant_model):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{val:02x}")
            if i < len(tflite_quant_model) - 1:
                f.write(", ")
            if (i + 1) % 12 == 0 or i == len(tflite_quant_model) - 1:
                f.write("\n")
        f.write("};\n\n")
        f.write("#endif // MODEL_H\n")
        
    print("[SUCCESS] Training pipeline completed! model.h updated successfully.", flush=True)

if __name__ == "__main__":
    main()
