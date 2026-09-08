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
    with open(r"siren_detection\model.h", 'r') as f:
        content = f.read()
        mean_match = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\};', content)
        std_match = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\};', content)
        mean = np.array([float(x.strip().replace('f', '')) for x in mean_match.group(1).split(',')])
        std = np.array([float(x.strip().replace('f', '')) for x in std_match.group(1).split(',')])
    return mean, std

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

def test_volume(file_path, mean, std, volume_multiplier):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    target_len = int(SAMPLE_RATE * DURATION)
    y = y[:target_len] if len(y) >= target_len else np.pad(y, (0, target_len - len(y)))
    
    # APPLY VOLUME CHANGE
    y = y * volume_multiplier
    
    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=0, fmax=4000, window='hamming', center=False)
    log_mel = np.log(S + 1e-9)
    feat = log_mel.T
    feat_scaled = (feat - mean[np.newaxis, :]) / std[np.newaxis, :]
    
    feat_input = np.expand_dims(feat_scaled, axis=0)
    feat_input = np.expand_dims(feat_input, axis=-1)
    
    probs = run_tflite_inference("siren_model_quant.tflite", feat_input)
    best_idx = np.argmax(probs)
    return CATEGORIES[best_idx], probs[best_idx]*100

def main():
    mean, std = get_scaler()
    file_path = r"C:\Users\ASUS\Videos\DATASET\POLICE\Suara Sirine Polisi.wav" # Clean file
    
    print("Testing POLICE file at different volumes:")
    for vol in [1.0, 0.5, 0.2, 0.05, 0.01]:
        pred, conf = test_volume(file_path, mean, std, vol)
        print(f"Volume x{vol}: Predicted {pred} ({conf:.1f}%)")

if __name__ == '__main__':
    main()
