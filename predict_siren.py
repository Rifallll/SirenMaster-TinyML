"""
Siren Classifier Inference Script
Allows predicting emergency vehicle siren types from a single WAV audio file.
Supports both Keras (.h5) and TensorFlow Lite (.tflite) model backends.

Usage:
  python predict_siren.py --file <path_to_wav> [--model <model_path>] [--backend <h5|tflite>]
"""

import os
import argparse
import numpy as np
import soundfile as sf
import librosa
import joblib

CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

def compute_features(y, sr):
    """Extracts 278 spectral and temporal features from a waveform chunk."""
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
    
    # 3. Extract Spectral Contrast (7 bands)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)
    
    # Concatenate features
    feature_vector = np.concatenate([
        mfcc_mean, mfcc_std,
        mfcc_delta_mean, mfcc_delta_std,
        mfcc_delta2_mean, mfcc_delta2_std,
        chroma_mean, chroma_std,
        contrast_mean, contrast_std
    ])
    
    return feature_vector

def extract_chunks_features(file_path):
    """
    Extracts features for sliding 2.0s windows with 50% overlap across the input file.
    Each window is scaled and processed by the model, then predictions are averaged.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    y, sr = sf.read(file_path)
    
    # Convert stereo to mono by averaging channels
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        
    # Resample to target 22050 Hz if needed
    TARGET_SR = 22050
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
        
    CHUNK_DUR = 2.0
    chunk_samples = int(CHUNK_DUR * TARGET_SR) # 44100
    hop_samples = chunk_samples // 2 # 50% overlap (1.0s hop) for smooth temporal voting
    
    # Slice chunks
    chunks = []
    for start in range(0, len(y) - chunk_samples + 1, hop_samples):
        chunk = y[start:start+chunk_samples]
        rms = np.sqrt(np.mean(chunk**2))
        if rms >= 0.0005:  # skip silent windows
            chunks.append(chunk)
            
    # Pad short files if the entire file is too short but not silent
    if len(chunks) == 0:
        if len(y) >= int(0.5 * TARGET_SR):
            rms = np.sqrt(np.mean(y**2))
            if rms >= 0.0005:
                padded = np.pad(y, (0, chunk_samples - len(y)), mode='constant')
                chunks.append(padded)
        else:
            raise ValueError("Audio file is too short (< 0.5s) or silent.")
            
    # Extract features for all chunks
    features_list = []
    for chunk in chunks:
        feat = compute_features(chunk, TARGET_SR)
        if feat is not None:
            features_list.append(feat)
            
    if len(features_list) == 0:
        raise ValueError("Could not extract valid features from the audio file.")
        
    return np.array(features_list)

def predict_h5(features, model_path):
    """Predicts class using the Keras .h5 model across all chunks and averages."""
    import tensorflow as tf
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Keras model file not found: {model_path}")
        
    model = tf.keras.models.load_model(model_path)
    
    # The CNN 1D expects shape (samples, features, channels) -> (num_chunks, 278, 1)
    features_input = features.reshape(len(features), -1, 1)
    
    all_probs = model.predict(features_input, verbose=0)
    probs = np.mean(all_probs, axis=0)
    return probs

def predict_tflite(features, model_path):
    """Predicts class using the TensorFlow Lite model across all chunks and averages."""
    import tensorflow as tf
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model file not found: {model_path}")
        
    # Load TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    all_probs = []
    for feat in features:
        # Reshape features to fit input details: usually [1, 278, 1]
        features_input = feat.reshape(input_details[0]['shape']).astype(np.float32)
        
        # Set tensor input
        interpreter.set_tensor(input_details[0]['index'], features_input)
        interpreter.invoke()
        
        # Get output probabilities
        probs = interpreter.get_tensor(output_details[0]['index'])[0]
        all_probs.append(probs)
        
    probs = np.mean(all_probs, axis=0)
    return probs

def main():
    parser = argparse.ArgumentParser(description="Predict Emergency Vehicle Siren type from audio WAV file.")
    parser.add_argument('--file', type=str, required=True, help="Path to the WAV audio file.")
    parser.add_argument('--scaler', type=str, default="siren_scaler.joblib", help="Path to saved scaler.")
    parser.add_argument('--model', type=str, default=None, help="Path to model file. Auto-detects by backend default.")
    parser.add_argument('--backend', type=str, choices=['h5', 'tflite'], default='tflite', 
                        help="Backend to use: 'h5' (Keras) or 'tflite' (TensorFlow Lite). Default is tflite.")
    
    args = parser.parse_args()
    
    # Resolve default model paths if not specified
    if args.model is None:
        if args.backend == 'h5':
            args.model = 'siren_classifier_model.h5'
        else:
            # Prefer quantized model if available, else fallback to standard TFLite
            if os.path.exists('siren_classifier_model_quant.tflite'):
                args.model = 'siren_classifier_model_quant.tflite'
            else:
                args.model = 'siren_classifier_model.tflite'
                
    print(f"[*] Processing file: {args.file}")
    print(f"[*] Backend: {args.backend.upper()} | Model: {args.model} | Scaler: {args.scaler}")
    
    # Load scaler
    if not os.path.exists(args.scaler):
        raise FileNotFoundError(f"Scaler joblib file not found: {args.scaler}. Please run training first.")
    scaler = joblib.load(args.scaler)
    
    # Extract & scale features
    features = extract_chunks_features(args.file)
    features_scaled = scaler.transform(features)
    
    # Run prediction based on selected backend
    if args.backend == 'h5':
        probs = predict_h5(features_scaled, args.model)
    else:
        probs = predict_tflite(features_scaled, args.model)
        
    # Print results
    print("\n" + "="*45)
    print(f" {'Category':<15} | {'Confidence (%)':<15}")
    print("="*45)
    
    best_idx = np.argmax(probs)
    for idx, (cat, prob) in enumerate(zip(CATEGORIES, probs)):
        marker = ">>>" if idx == best_idx else "   "
        print(f"{marker} {cat:<12} | {prob*100:>8.2f}%")
        
    print("="*45)
    print(f"Result: Vehicle is classified as {CATEGORIES[best_idx]} ({probs[best_idx]*100:.2f}% confidence)\n")

if __name__ == '__main__':
    main()
