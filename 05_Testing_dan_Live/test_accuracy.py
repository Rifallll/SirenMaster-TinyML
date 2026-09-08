import os
import random
import librosa
import numpy as np
import tensorflow as tf
import re

SAMPLE_RATE = 8000
N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40
DURATION = 1.024
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'POLICE', 'NORMAL']

def get_scaler():
    mean, std = [], []
    with open(r"siren_detection\model.h", 'r') as f:
        content = f.read()
        mean_match = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\};', content)
        std_match = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\};', content)
        mean = np.array([float(x.strip().replace('f', '')) for x in mean_match.group(1).split(',')])
        std = np.array([float(x.strip().replace('f', '')) for x in std_match.group(1).split(',')])
    return mean, std

def extract_features(file_path, mean, std):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    target_len = int(SAMPLE_RATE * DURATION)
    
    if len(y) > target_len:
        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        max_idx = np.argmax(rms) * HOP_LENGTH
        start = max(0, max_idx - target_len // 2)
        if start + target_len > len(y):
            start = len(y) - target_len
        y = y[start:start+target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
        
    S = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=0, fmax=4000,
        window='hamming', center=False
    )
    log_mel = np.log(S + 1e-9)
    feat = log_mel.T # shape: (63, 40)
    feat_scaled = (feat - mean[np.newaxis, :]) / std[np.newaxis, :]
    
    # Reshape for CNN
    feat_input = np.expand_dims(feat_scaled, axis=0) # (1, 63, 40)
    feat_input = np.expand_dims(feat_input, axis=-1) # (1, 63, 40, 1)
    return feat_input

def run_tflite_inference(model_path, feat_input):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
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

def main():
    base_dir = r"C:\Users\ASUS\Videos\DATASET"
    model_path = r"siren_model_quant.tflite"
    mean, std = get_scaler()
    
    for cat in CATEGORIES:
        cat_dir = os.path.join(base_dir, cat)
        files = [f for f in os.listdir(cat_dir) if f.endswith('.wav') and 'augment' not in f]
        if not files: continue
        
        sample_file = os.path.join(cat_dir, random.choice(files))
        feat = extract_features(sample_file, mean, std)
        probs = run_tflite_inference(model_path, feat)
        
        print(f"--- File: {cat}/{os.path.basename(sample_file)} ---")
        for i, c in enumerate(CATEGORIES):
            print(f"  {c}: {probs[i]*100:.2f}%")
        print()

if __name__ == '__main__':
    main()
