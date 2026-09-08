import numpy as np
import tensorflow as tf
import librosa
import os
import glob

# Load scaler
scaler_data = np.load(r"C:\Users\ASUS\Videos\DATASET\siren_scaler.npz")
global_mean = scaler_data['global_mean']
global_std = scaler_data['global_std']

SAMPLE_RATE = 8000
DURATION = 4.0
N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40

def extract_melspec(y):
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
        power   = np.abs(fft_out) ** 2
        mel_e   = np.dot(mel_fb, power)
        log_mel_frames.append(np.log(mel_e + 1e-9))

    return np.array(log_mel_frames, dtype=np.float32)

def predict_file(model_path, file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    feat = extract_melspec(y)
    
    # Scale features
    feat_s = (feat - global_mean) / global_std
    feat_s = feat_s[np.newaxis, ..., np.newaxis] # Shape (1, 249, 40, 1)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    is_int8 = input_details[0]['dtype'] == np.int8

    if is_int8:
        sc_in, zp_in = input_details[0]['quantization']
        feat_q = (feat_s / sc_in) + zp_in
        feat_q = np.round(feat_q).astype(np.int8)
        feat_q = np.clip(feat_q, -128, 127)
        interpreter.set_tensor(input_details[0]['index'], feat_q)
    else:
        interpreter.set_tensor(input_details[0]['index'], feat_s.astype(np.float32))

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    if is_int8:
        sc_out, zp_out = output_details[0]['quantization']
        probs = (output.astype(np.float32) - zp_out) * sc_out
    else:
        probs = output
        
    return probs

# Test files from each folder
categories = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
model_path = r"C:\Users\ASUS\Videos\DATASET\siren_model_quant.tflite"

for cat in categories:
    folder = os.path.join(r"C:\Users\ASUS\Videos\DATASET", cat)
    files = glob.glob(os.path.join(folder, "*.wav"))
    if len(files) > 0:
        test_file = files[0]
        probs = predict_file(model_path, test_file)
        pred_idx = np.argmax(probs)
        print(f"File: {cat}/{os.path.basename(test_file)}")
        print(f"  Probs: {['%.2f%%' % (p*100) for p in probs]}")
        print(f"  Result: {categories[pred_idx]} (idx: {pred_idx})")
        print("-" * 50)
