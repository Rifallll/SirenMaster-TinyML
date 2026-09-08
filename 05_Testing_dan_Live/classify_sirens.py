"""
Siren Classifier - Emergency Vehicle Siren Sound Classification Script
This script implements a complete pipeline to classify WAV files into four categories:
1. AMBULANCE
2. FIRETRUCK
3. POLICE
4. NORMAL

Pipeline steps:
1. Load WAV files recursively from 4 folders.
2. Parallel feature extraction using soundfile and librosa (MFCC, Chroma, Spectral Contrast).
3. Cache extracted features to disk to make subsequent runs instantaneous.
4. Stratified 80/20 train-test split.
5. Standard scaling of features (critical for neural networks and SMOTE's distance metrics).
6. Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) on the training split only.
7. Build and train a 1D CNN or MLP model.
8. Evaluate the model (Accuracy, Classification Report, Confusion Matrix).
9. Save the trained model to .h5 format, the evaluation report to CSV, and the plots to PNG.
"""

import os
import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ignore warnings for clean progress logs
warnings.filterwarnings('ignore')

# Heavy machine learning/neural network imports are moved inside functions 
# to prevent subprocesses from loading them when spawning parallel workers.


# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = r"C:\Users\ASUS\Videos\DATASET"
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
CACHE_FILE = 'siren_features_cache_v2.npz'
MODEL_TYPE = 'cnn1d'  # Options: 'cnn1d' or 'mlp'
MODEL_SAVE_PATH = 'siren_classifier_model.h5'
CSV_REPORT_PATH = 'evaluation_report.csv'
MAX_WORKERS = 12       # Number of CPU cores for parallel feature extraction

# ==========================================
# 1. HELPERS & FEATURE EXTRACTION
# ==========================================
def get_improved_group_name(file_path):
    """
    Extracts the root recording group name by stripping off suffixes 
    (_original, _loud, _noise_light, _shift, _segXX, (X), -[AudioTrimmer.com])
    to prevent files from the same recording being split across train and test sets.
    """
    base = os.path.splitext(os.path.basename(file_path))[0]
    match = re.match(r'^([a-zA-Z]+_\d+)', base)
    if match:
        return match.group(1)
    cleaned = base
    cleaned = re.sub(r'\s*\(\d+\)$', '', cleaned)
    cleaned = re.sub(r'_(original|loud|noise_light|shift|seg\d+|_seg\d+)$', '', cleaned)
    cleaned = cleaned.replace('-[AudioTrimmer.com]', '').strip()
    return cleaned

def extract_features_from_waveform(y, sr):
    """
    Extracts 278 features (MFCC, MFCC Delta, MFCC Delta-Delta, Chroma, Spectral Contrast) 
    from a raw waveform array y.
    """
    try:
        # Ensure audio signal has at least 8192 samples (required for delta/delta-delta calculations)
        if len(y) < 8192:
            y = np.pad(y, (0, 8192 - len(y)), mode='constant')
            
        # 1. Extract MFCC (40 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # MFCC Delta (velocity)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta_std = np.std(mfcc_delta, axis=1)
        
        # MFCC Delta-Delta (acceleration)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
        mfcc_delta2_std = np.std(mfcc_delta2, axis=1)
        
        # 2. Extract Chroma STFT (12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # 3. Extract Spectral Contrast (7 bands by default)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)
        
        # Concatenate mean and std of all features into a 1D vector (278 features)
        feature_vector = np.concatenate([
            mfcc_mean, mfcc_std,
            mfcc_delta_mean, mfcc_delta_std,
            mfcc_delta2_mean, mfcc_delta2_std,
            chroma_mean, chroma_std,
            contrast_mean, contrast_std
        ])
        return feature_vector
    except Exception:
        return None

def process_file_chunks(task):
    """
    Parallel worker that processes a single file. Loads, resamples, slices into 
    fixed 2-second chunks, applies augmentations if in training set, and extracts features.
    """
    file_path, label, is_training = task
    import soundfile as sf
    import librosa
    import numpy as np
    
    features_list = []
    try:
        # Determine maximum duration we need to read to save time and RAM
        if is_training:
            if label == 0:        # AMBULANCE: 8 chunks with 50% overlap of 2s chunks starts at 7s + 2s = 9s
                max_seconds = 9.0
            elif label in [2, 3]: # POLICE, NORMAL: 4 chunks starts at 3s + 2s = 5s
                max_seconds = 5.0
            else:                 # FIRETRUCK: 1 chunk = 2s
                max_seconds = 2.0
        else:
            max_seconds = 4.0     # Test split: 2 chunks = 4s
            
        info = sf.info(file_path)
        frames_to_read = int(max_seconds * info.samplerate)
        y, sr = sf.read(file_path, frames=frames_to_read)
        
        # Convert stereo to mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        # Resample to 22050 Hz using faster kaiser_fast type
        TARGET_SR = 22050
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type='kaiser_fast')
            sr = TARGET_SR
            
        CHUNK_DUR = 2.0
        chunk_samples = int(CHUNK_DUR * TARGET_SR) # 44100
        
        if is_training:
            # Class-specific density settings to naturally balance classes
            if label == 0:    # AMBULANCE (minority) -> 50% overlap, max 8 chunks, 2 augmented variants
                hop_samples = chunk_samples // 2
                max_chunks = 8
                num_augmentations = 2
            elif label == 2:  # POLICE -> 50% overlap, max 4 chunks, 1 augmented variant
                hop_samples = chunk_samples // 2
                max_chunks = 4
                num_augmentations = 1
            elif label == 3:  # NORMAL -> 50% overlap, max 4 chunks, 1 augmented variant
                hop_samples = chunk_samples // 2
                max_chunks = 4
                num_augmentations = 1
            else:             # FIRETRUCK (majority) -> no overlap, max 1 chunk, no augmentations
                hop_samples = chunk_samples
                max_chunks = 1
                num_augmentations = 0
        else:
            # Test split chunks are clean, non-overlapping, and max 2 per file
            hop_samples = chunk_samples
            max_chunks = 2
            num_augmentations = 0
            
        # Slicing into chunks
        chunks = []
        for start in range(0, len(y) - chunk_samples + 1, hop_samples):
            chunk = y[start:start+chunk_samples]
            rms = np.sqrt(np.mean(chunk**2))
            if rms >= 0.0005:  # skip silent chunks
                chunks.append(chunk)
                
        # Limit max chunks
        if len(chunks) > max_chunks:
            indices = np.linspace(0, len(chunks) - 1, max_chunks, dtype=int)
            chunks = [chunks[i] for i in indices]
            
        # Pad short files if they are not silent
        if len(chunks) == 0:
            if len(y) >= int(0.5 * TARGET_SR):
                rms = np.sqrt(np.mean(y**2))
                if rms >= 0.0005:
                    padded = np.pad(y, (0, chunk_samples - len(y)), mode='constant')
                    chunks.append(padded)
                    
        # Feature extraction
        for chunk in chunks:
            # 1. Clean chunk features
            feat = extract_features_from_waveform(chunk, TARGET_SR)
            if feat is not None:
                features_list.append(feat)
                
            # 2. Waveform augmentations (only during training for minority classes)
            if is_training and num_augmentations > 0:
                for _ in range(num_augmentations):
                    # Random volume gain (volume variations)
                    gain = np.random.uniform(0.6, 1.4)
                    y_aug = chunk * gain
                    
                    # Random background white noise
                    noise_amp = np.random.uniform(0.002, 0.012)
                    noise = np.random.normal(0, 1, len(y_aug))
                    y_aug = y_aug + noise_amp * noise
                    
                    # Random time shift
                    shift = np.random.randint(500, 2000)
                    y_aug = np.roll(y_aug, shift)
                    
                    feat_aug = extract_features_from_waveform(y_aug, TARGET_SR)
                    if feat_aug is not None:
                        features_list.append(feat_aug)
    except Exception:
        pass
        
    return features_list

# ==========================================
# 2. LOAD DATASET AND EXTRACT FEATURES (SPLIT FIRST)
# ==========================================
def get_dataset_split():
    """
    Loads, splits, chunks, and processes the dataset.
    Performs Group Split at the file-level BEFORE feature extraction to ensure 0% data leakage.
    Saves/loads results to CACHE_FILE.
    """
    if os.path.exists(CACHE_FILE):
        print(f"[PROGRESS] Found cached features at {CACHE_FILE}. Loading cache...")
        data = np.load(CACHE_FILE, allow_pickle=True)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        print(f"Loaded from cache successfully:")
        print(f"  Train set: {len(X_train)} samples")
        print(f"  Test set:  {len(X_test)} samples")
        return X_train, y_train, X_test, y_test
        
    print("[PROGRESS] Scanning dataset folders for WAV files...")
    file_paths = []
    labels = []
    groups = []
    
    for label_idx, cat in enumerate(CATEGORIES):
        cat_dir = os.path.join(DATASET_PATH, cat)
        if not os.path.exists(cat_dir):
            print(f"Warning: Category folder {cat_dir} does not exist!")
            continue
            
        cat_files = []
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    cat_files.append(os.path.join(root, file))
                    
        print(f"  Category '{cat}': found {len(cat_files)} WAV files.")
        file_paths.extend(cat_files)
        labels.extend([label_idx] * len(cat_files))
        groups.extend([get_improved_group_name(fp) for fp in cat_files])
        
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    groups = np.array(groups)
    
    if len(file_paths) == 0:
        raise ValueError(f"No WAV files found in dataset path: {DATASET_PATH}")
        
    # Perform Group Split at the recording level BEFORE feature extraction
    print("[PROGRESS] Splitting dataset into train (80%) and test (20%) sets based on original recording groups...")
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(file_paths, labels, groups=groups))
    
    train_paths, test_paths = file_paths[train_idx], file_paths[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    
    print(f"Total unique groups in dataset: {len(np.unique(groups))}")
    print(f"Unique groups in Train set: {len(np.unique(groups[train_idx]))}")
    print(f"Unique groups in Test set: {len(np.unique(groups[test_idx]))}")
    
    # Build parallel feature extraction tasks
    # Task format: (file_path, label, is_training)
    train_tasks = [(fp, lbl, True) for fp, lbl in zip(train_paths, train_labels)]
    test_tasks = [(fp, lbl, False) for fp, lbl in zip(test_paths, test_labels)]
    
    print(f"[PROGRESS] Starting parallel extraction on Train set ({len(train_tasks)} files) using {MAX_WORKERS} workers...")
    X_train = []
    y_train = []
    
    start_time = time.time()
    processed_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file_chunks, t): t for t in train_tasks}
        for fut in as_completed(futures):
            _, lbl, _ = futures[fut]
            chunk_feats = fut.result()
            if chunk_feats:
                X_train.extend(chunk_feats)
                y_train.extend([lbl] * len(chunk_feats))
            processed_count += 1
            if processed_count % 500 == 0 or processed_count == len(train_tasks):
                elapsed = time.time() - start_time
                print(f"  Train: processed {processed_count}/{len(train_tasks)} files ({processed_count/len(train_tasks)*100:.1f}%) | Elapsed: {elapsed:.1f}s")
                
    print(f"[PROGRESS] Starting parallel extraction on Test set ({len(test_tasks)} files) using {MAX_WORKERS} workers...")
    X_test = []
    y_test = []
    
    start_time = time.time()
    processed_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file_chunks, t): t for t in test_tasks}
        for fut in as_completed(futures):
            _, lbl, _ = futures[fut]
            chunk_feats = fut.result()
            if chunk_feats:
                X_test.extend(chunk_feats)
                y_test.extend([lbl] * len(chunk_feats))
            processed_count += 1
            if processed_count % 500 == 0 or processed_count == len(test_tasks):
                elapsed = time.time() - start_time
                print(f"  Test: processed {processed_count}/{len(test_tasks)} files ({processed_count/len(test_tasks)*100:.1f}%) | Elapsed: {elapsed:.1f}s")
                
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"[PROGRESS] Saving extracted features to cache file: {CACHE_FILE}")
    np.savez_compressed(CACHE_FILE, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    
    return X_train, y_train, X_test, y_test

# ==========================================
# 3. BUILD MODEL ARCHITECTURES
# ==========================================
def build_mlp_model(input_shape, num_classes):
    """
    Builds a standard Multi-Layer Perceptron (MLP) model.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    
    model = Sequential([
        Input(shape=(input_shape,)),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_cnn1d_model(input_shape, num_classes):
    """
    Builds a 1D Convolutional Neural Network (CNN 1D) model.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Conv1D, MaxPooling1D, Flatten
    
    model = Sequential([
        Input(shape=(input_shape, 1)),
        
        Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(num_classes, activation='softmax')
    ])
    return model

# ==========================================
# 4. MAIN PIPELINE EXECUTION
# ==========================================
def main():
    # Local imports to prevent subprocesses from loading heavy packages when spawning
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from imblearn.combine import SMOTETomek
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # --- Step 1 & 2: Load, Split, Chunk, and Extract Features Cleanly ---
    X_train, y_train, X_test, y_test = get_dataset_split()
    
    # Print initial class distribution
    print("Class distribution in Train set:")
    for idx, cat in enumerate(CATEGORIES):
        count = np.sum(y_train == idx)
        print(f"  {cat}: {count} samples ({count/len(y_train)*100:.1f}%)")
        
    print("Class distribution in Test set:")
    for idx, cat in enumerate(CATEGORIES):
        count = np.sum(y_test == idx)
        print(f"  {cat}: {count} samples ({count/len(y_test)*100:.1f}%)")
        
    # --- Step 3: Feature Scaling ---
    print("[PROGRESS] Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler so it can be reused during inference
    import joblib
    joblib.dump(scaler, 'siren_scaler.joblib')
    print("  Saved scaler to 'siren_scaler.joblib'.")
    
    # --- Step 4: Handle Class Imbalance with SMOTETomek ---
    print("[PROGRESS] Applying SMOTETomek to handle class imbalance and clean boundaries...")
    smote_tomek = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote_tomek.fit_resample(X_train_scaled, y_train)
    
    # Print balanced class distribution
    print("Class distribution in Train set after SMOTETomek:")
    for idx, cat in enumerate(CATEGORIES):
        count = np.sum(y_train_res == idx)
        print(f"  {cat}: {count} samples")
        
    # --- Step 5: Model Selection and Data Reshaping ---
    num_classes = len(CATEGORIES)
    input_dim = X_train_res.shape[1]
    
    if MODEL_TYPE.lower() == 'cnn1d':
        print(f"[PROGRESS] Initializing CNN 1D Model (input dimension: {input_dim})...")
        # Reshape input vectors to (samples, features, channels) for 1D CNN
        X_train_final = X_train_res.reshape(-1, input_dim, 1)
        X_test_final = X_test_scaled.reshape(-1, input_dim, 1)
        model = build_cnn1d_model(input_dim, num_classes)
    else:
        print(f"[PROGRESS] Initializing MLP Model (input dimension: {input_dim})...")
        X_train_final = X_train_res
        X_test_final = X_test_scaled
        model = build_mlp_model(input_dim, num_classes)
        
    model.summary()
    
    # Compile model with Adam optimizer and sparse categorical crossentropy
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # --- Step 6: Training ---
    print(f"[PROGRESS] Training model '{MODEL_TYPE}' for up to 50 epochs...")
    
    # Callbacks to optimize training process
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )
    
    start_train_time = time.time()
    history = model.fit(
        X_train_final, y_train_res,
        epochs=50,
        batch_size=64,
        validation_data=(X_test_final, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    train_duration = time.time() - start_train_time
    print(f"[PROGRESS] Training completed in {train_duration:.1f} seconds.")
    
    # --- Step 7: Model Evaluation ---
    print("[PROGRESS] Evaluating model on the test set...")
    
    # Predict classes
    y_pred_probs = model.predict(X_test_final)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate test accuracy
    test_loss, test_acc = model.evaluate(X_test_final, y_test, verbose=0)
    print(f"\n=========================================")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"=========================================\n")
    
    # Generate classification report
    report_text = classification_report(y_test, y_pred, target_names=CATEGORIES)
    print("Classification Report:")
    print(report_text)
    
    # Save classification report to CSV
    report_dict = classification_report(y_test, y_pred, target_names=CATEGORIES, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(CSV_REPORT_PATH, index=True)
    print(f"Saved classification report to '{CSV_REPORT_PATH}'.")
    
    # --- Step 8: Save Model ---
    model.save(MODEL_SAVE_PATH)
    print(f"Saved trained model to '{MODEL_SAVE_PATH}'.")
    
    # --- Step 9: Visualization ---
    print("[PROGRESS] Generating and saving evaluation plots...")
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f'Confusion Matrix ({MODEL_TYPE.upper()})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("  Saved confusion matrix plot to 'confusion_matrix.png'.")
    
    # Plot 2: Training History
    plt.figure(figsize=(12, 4))
    
    # Accuracy history
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Loss history
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    print("  Saved training history plot to 'training_history.png'.")
    
    # Close plots to free up memory
    plt.close('all')
        
    print("\n[SUCCESS] Pipeline execution finished successfully!")

if __name__ == '__main__':
    main()
