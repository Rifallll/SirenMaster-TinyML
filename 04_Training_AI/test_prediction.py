"""
test_prediction.py
==================
Skrip verifikasi untuk memprediksi beberapa sampel dari folder AMBULANCE, FIRETRUCK,
dan POLICE menggunakan model TFLite terkuantisasi hasil training terakhir.
Ini membuktikan secara ilmiah apakah model salah memetakan indeks kelas atau tidak.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, glob, librosa, numpy as np
import tensorflow as tf

ROOT = r"C:\Users\ASUS\Videos\DATASET"
TFLITE_PATH = os.path.join(ROOT, "siren_model_quant.tflite")
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Cek kuantisasi
in_scale, in_zero = input_details[0]['quantization']
out_scale, out_zero = output_details[0]['quantization']
is_quant = (in_scale != 0.0)

# Load Scaler
scaler = np.load(os.path.join(ROOT, "siren_scaler.npz"))
global_mean = scaler['global_mean']
global_std = scaler['global_std']

def predict_file(filepath):
    # Load audio
    y, sr = librosa.load(filepath, sr=8000, duration=4.0)
    if len(y) < 32000:
        y = np.pad(y, (0, 32000 - len(y)))
    else:
        y = y[:32000]

    # Ekstraksi Mel-Spectrogram (sama persis dengan training & ESP32)
    n_frames = (32000 - 256) // 128 + 1
    hamming = np.hamming(256)
    mel_fb = librosa.filters.mel(sr=8000, n_fft=256, n_mels=40, fmin=0, fmax=4000)

    log_mel_frames = []
    for frame in range(n_frames):
        start = frame * 128
        fd = y[start:start+256].copy()
        fft_out = np.fft.rfft(fd * hamming, n=256)
        power = np.abs(fft_out) ** 2
        mel_e = np.dot(mel_fb, power)
        log_mel_frames.append(np.log(mel_e + 1e-9))

    feat = np.array(log_mel_frames, dtype=np.float32) # (249, 40)
    
    # Standarisasi (Z-score) menggunakan nilai global training scaler
    feat = (feat - global_mean) / global_std

    # Reshape untuk input model (1, 249, 40, 1)
    input_data = np.expand_dims(feat, axis=(0, -1))

    # Kuantisasi input jika model INT8
    if is_quant:
        input_data = np.round(input_data / in_scale) + in_zero
        input_data = np.clip(input_data, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Dekuantisasi output
    if is_quant:
        output_data = (output_data.astype(np.float32) - out_zero) * out_scale

    probs = output_data[0]
    best_idx = np.argmax(probs)
    return CATEGORIES[best_idx], probs[best_idx], probs

print("=" * 75)
print("  UJI PREDIKSI MODEL TFLITE PADA FILE DATASET ASLI")
print("=" * 75)

for cat in ['AMBULANCE', 'FIRETRUCK', 'POLICE']:
    folder = os.path.join(ROOT, cat)
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))[:3] # Ambil 3 file pertama
    print(f"\n[📂 KELAS ASLI: {cat}]")
    for f in files:
        pred_label, conf, probs = predict_file(f)
        prob_str = ", ".join([f"{c}: {p*100:.1f}%" for c, p in zip(CATEGORIES, probs)])
        print(f"  • File: {os.path.basename(f)}")
        print(f"    -> PREDIKSI AI: {pred_label} ({conf*100:.1f}%)")
        print(f"    -> Detail Prob: [{prob_str}]")
print("=" * 75)
