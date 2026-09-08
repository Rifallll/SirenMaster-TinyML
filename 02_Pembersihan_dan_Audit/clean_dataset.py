import os
import shutil
import numpy as np
import librosa
import tensorflow as tf
import re

SAMPLE_RATE = 8000
N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40
DURATION = 1.024
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

def get_scaler():
    with open(r"siren_detection\model.h", 'r') as f:
        content = f.read()
        mean_match = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\};', content)
        std_match = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\};', content)
        mean = np.array([float(x.strip().replace('f', '')) for x in mean_match.group(1).split(',')])
        std = np.array([float(x.strip().replace('f', '')) for x in std_match.group(1).split(',')])
    return mean, std

def run_tflite_inference(interpreter, input_details, output_details, feat_input):
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        feat_input = np.round(feat_input / scale + zero_point).astype(np.int8)
        
    interpreter.set_tensor(input_details[0]['index'], feat_input)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        probs = (output_data.astype(np.float32) - zero_point) * scale
    else:
        probs = output_data[0]
    return probs[0]

def analyze_and_clean_dataset():
    mean, std = get_scaler()
    interpreter = tf.lite.Interpreter(model_path="siren_model_quant.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    base_dir = r"C:\Users\ASUS\Videos\DATASET"
    trash_dir = os.path.join(base_dir, "DATASET_TRASH")
    os.makedirs(trash_dir, exist_ok=True)
    
    for target_cat, avoid_cat in [('POLICE', 'AMBULANCE'), ('AMBULANCE', 'POLICE')]:
        cat_dir = os.path.join(base_dir, target_cat)
        trash_cat_dir = os.path.join(trash_dir, target_cat)
        os.makedirs(trash_cat_dir, exist_ok=True)
        
        files = [f for f in os.listdir(cat_dir) if f.endswith('.wav') and not f.startswith('aug_')]
        avoid_idx = CATEGORIES.index(avoid_cat)
        
        ambiguous_count = 0
        print(f"\nMemindai {len(files)} file {target_cat} untuk mencari kemiripan dengan {avoid_cat}...")
        
        for i, f in enumerate(files):
            file_path = os.path.join(cat_dir, f)
            try:
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue
                
            target_len = int(SAMPLE_RATE * DURATION)
            y = y[:target_len] if len(y) >= target_len else np.pad(y, (0, target_len - len(y)))
            
            max_val = np.max(np.abs(y))
            if max_val > 1e-6:
                y = y * min(1.0 / max_val, 10.0)
            
            S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=0, fmax=4000, window='hamming', center=False)
            log_mel = np.log(S + 1e-9)
            feat_scaled = (log_mel.T - mean[np.newaxis, :]) / std[np.newaxis, :]
            
            feat_input = np.expand_dims(np.expand_dims(feat_scaled, axis=0), axis=-1)
            probs = run_tflite_inference(interpreter, input_details, output_details, feat_input)
            
            if probs[avoid_idx] > 0.3:
                # Memindahkan file yang membingungkan ke folder TRASH
                dest_path = os.path.join(trash_cat_dir, f)
                shutil.move(file_path, dest_path)
                ambiguous_count += 1
                
        print(f"-> Berhasil memindahkan {ambiguous_count} file yang meragukan dari {target_cat} ke folder DATASET_TRASH.")

if __name__ == '__main__':
    analyze_and_clean_dataset()
