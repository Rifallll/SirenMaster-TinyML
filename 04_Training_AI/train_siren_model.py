import os
import re
import sys
import numpy as np
import librosa
import math
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set encoding for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# --- CONFIGURATION ---
DATASET_PATH = r"C:\Users\ASUS\Videos\DATASET"
OUTPUT_HEADER = r"C:\Users\ASUS\Videos\DATASET\sirenmaster_main\model.h"
CACHE_PATH = "siren_13x63_features_cache.npz"

SAMPLE_RATE = 8000
N_FFT = 256
HOP_LENGTH = 128
N_MFCC = 13
N_MELS = 40
DURATION = 1.024  # 8192 samples
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

def extract_mfcc(y, sr=SAMPLE_RATE):
    """Extracts MFCCs using Hamming window to match C++ DSP exactly."""
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
        
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=0, fmax=4000,
        window='hamming'
    )
    return mfccs.T  # shape: (63, 13)

def process_single_file(fp, label, is_training):
    """Processes a single file, extracts features, and applies augmentations."""
    X_file = []
    y_file = []
    target_len = int(SAMPLE_RATE * DURATION)
    try:
        y, sr = librosa.load(fp, sr=SAMPLE_RATE)
        
        if is_training:
            hop = target_len // 2
        else:
            hop = target_len
            
        for start in range(0, len(y) - target_len + 1, hop):
            chunk = y[start:start+target_len]
            rms = np.sqrt(np.mean(chunk**2))
            if rms < 0.001:
                continue
                
            # Clean
            feat = extract_mfcc(chunk)
            X_file.append(feat)
            y_file.append(label)
            
            if is_training:
                # 1. Noise
                noise = np.random.normal(0, 0.003, len(chunk))
                feat_noise = extract_mfcc(chunk + noise)
                X_file.append(feat_noise)
                y_file.append(label)
                
                # 2. Pitch Shift +1.5
                try:
                    shifted = librosa.effects.pitch_shift(chunk, sr=SAMPLE_RATE, n_steps=1.5)
                    X_file.append(extract_mfcc(shifted))
                    y_file.append(label)
                except:
                    pass
                    
                # 3. Pitch Shift -1.5
                try:
                    shifted = librosa.effects.pitch_shift(chunk, sr=SAMPLE_RATE, n_steps=-1.5)
                    X_file.append(extract_mfcc(shifted))
                    y_file.append(label)
                except:
                    pass
                    
                # 4. Random Shift
                shift = np.random.randint(500, 2000)
                rolled = np.roll(chunk, shift)
                X_file.append(extract_mfcc(rolled))
                y_file.append(label)
                
        # Pad short files
        if len(y) < target_len and len(y) >= target_len // 2:
            rms = np.sqrt(np.mean(y**2))
            if rms >= 0.001:
                padded = np.pad(y, (0, target_len - len(y)), mode='constant')
                feat = extract_mfcc(padded)
                X_file.append(feat)
                y_file.append(label)
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
        cat_dir = os.path.join(DATASET_PATH, cat)
        if not os.path.exists(cat_dir):
            print(f"  [WARNING] Folder {cat_dir} does not exist!", flush=True)
            continue
            
        cat_files = []
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    cat_files.append(os.path.join(root, file))
                    
        print(f"  Category '{cat}': found {len(cat_files)} files.", flush=True)
        file_paths.extend(cat_files)
        labels.extend([idx] * len(cat_files))
        groups.extend([get_improved_group_name(fp) for fp in cat_files])
        
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    groups = np.array(groups)
    
    if len(file_paths) == 0:
        raise ValueError(f"No WAV files found at: {DATASET_PATH}")
        
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
    global_mean = np.mean(X_train, axis=(0, 1))  # (13,)
    global_std = np.std(X_train, axis=(0, 1))    # (13,)
    global_std[global_std < 1e-6] = 1.0
    
    # Scale train and test datasets
    X_train_scaled = (X_train - global_mean[np.newaxis, np.newaxis, :]) / global_std[np.newaxis, np.newaxis, :]
    X_test_scaled = (X_test - global_mean[np.newaxis, np.newaxis, :]) / global_std[np.newaxis, np.newaxis, :]
    
    print("  Mean values per MFCC coefficient:", np.round(global_mean, 4), flush=True)
    print("  Std values per MFCC coefficient:", np.round(global_std, 4), flush=True)
    
    # 5. Build and Train Model
    print("\n[*] Importing TensorFlow and Keras...", flush=True)
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    print("\n[*] Building 1D CNN model...", flush=True)
    model = models.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(len(CATEGORIES), activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # Class weights to help minority classes
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    
    print("\n[*] Training CNN model...", flush=True)
    model.fit(
        X_train_scaled, y_train,
        epochs=40,
        batch_size=256,
        validation_data=(X_test_scaled, y_test),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Evaluate Model
    print("\n[*] Evaluating model on unseen test split...", flush=True)
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nConfusion Matrix:", flush=True)
    print(confusion_matrix(y_test, y_pred), flush=True)
    print("\nClassification Report:", flush=True)
    print(classification_report(y_test, y_pred, target_names=CATEGORIES), flush=True)
    
    # 7. Quantization to INT8
    print("\n[*] Quantizing model to INT8 for TFLite Micro deployment...", flush=True)
    def representative_data_gen():
        for i in range(min(300, len(X_train_scaled))):
            yield [X_train_scaled[i:i+1].astype(np.float32)]
            
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    
    # Save quantized TFLite binary to disk for local testing
    with open('siren_model_quant.tflite', 'wb') as f_tflite:
        f_tflite.write(tflite_quant_model)
    print("  Quantized model saved to siren_model_quant.tflite", flush=True)
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
        
        # DCT Matrix
        f.write(f"const float DCT_MATRIX[{N_MFCC}][{N_MELS}] PROGMEM = {{\n")
        for c in range(N_MFCC):
            f.write("  {")
            row = []
            for m in range(N_MELS):
                if c == 0:
                    row.append(1.0 / math.sqrt(N_MELS))
                else:
                    row.append(math.sqrt(2.0 / N_MELS) * math.cos(math.pi * c * (m + 0.5) / N_MELS))
            f.write(", ".join([f"{val:.8f}f" for val in row]))
            f.write("}")
            if c < N_MFCC - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")
        
        # Scaling parameters (MFCC_MEAN and MFCC_STD)
        f.write(f"const float MFCC_MEAN[{N_MFCC}] PROGMEM = {{\n  ")
        f.write(", ".join([f"{val:.8f}f" for val in global_mean]))
        f.write("\n};\n\n")
        
        f.write(f"const float MFCC_STD[{N_MFCC}] PROGMEM = {{\n  ")
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
