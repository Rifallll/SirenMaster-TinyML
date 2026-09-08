"""
audit_10_sampel_ujian.py
========================
Mengecek volume audio dan akurasi prediksi model AI (TFLite INT8)
terhadap 10 sampel ujian resmi yang ada di putar_sirene.py.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, librosa, numpy as np
import tensorflow as tf

ROOT = r"C:\Users\ASUS\Videos\DATASET"
TFLITE = os.path.join(ROOT, "04_Training_AI", "siren_model_quant.tflite")
if not os.path.exists(TFLITE):
    TFLITE = os.path.join(ROOT, "siren_model_quant.tflite")
SCALER = os.path.join(ROOT, "04_Training_AI", "siren_scaler.npz")

CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

interp = tf.lite.Interpreter(model_path=TFLITE)
interp.allocate_tensors()
in_idx = interp.get_input_details()[0]['index']
out_idx = interp.get_output_details()[0]['index']
in_scale, in_zero = interp.get_input_details()[0]['quantization']
out_scale, out_zero = interp.get_output_details()[0]['quantization']
is_quant = (in_scale != 0.0)

scaler = np.load(SCALER)
g_mean = scaler['global_mean']
g_std = scaler['global_std']

EXAM_10_SAMPLES = [
    ("AMBULANCE #1", os.path.join(ROOT, "AMBULANCE", "ambulance_0008_seg10.wav"), "AMBULANCE"),
    ("DAMKAR #1", os.path.join(ROOT, "FIRETRUCK", "fire_0002_seg10.wav"), "FIRETRUCK"),
    ("POLISI #1", os.path.join(ROOT, "POLICE", "police_0001_seg10.wav"), "POLICE"),
    ("AMBULANCE #2", os.path.join(ROOT, "AMBULANCE", "ambulance_0008_seg15.wav"), "AMBULANCE"),
    ("DAMKAR #2", os.path.join(ROOT, "FIRETRUCK", "fire_0003_seg15.wav"), "FIRETRUCK"),
    ("POLISI #2", os.path.join(ROOT, "POLICE", "police_0001_seg15.wav"), "POLICE"),
    ("AMBULANCE #3", os.path.join(ROOT, "AMBULANCE", "ambulance_0008_seg20.wav"), "AMBULANCE"),
    ("DAMKAR #3 (Coaster Rumbler)", os.path.join(ROOT, "FIRETRUCK", "fire_0010_seg10.wav"), "FIRETRUCK"),
    ("POLISI #3", os.path.join(ROOT, "POLICE", "police_0001_seg20.wav"), "POLICE"),
    ("DAMKAR #4", os.path.join(ROOT, "FIRETRUCK", "fire_0020_seg10.wav"), "FIRETRUCK"),
]

def extract_feat(filepath):
    y, sr = librosa.load(filepath, sr=8000, duration=4.0)
    rms = np.sqrt(np.mean(y**2))
    max_amp = np.max(np.abs(y))
    
    if len(y) < 32000:
        y = np.pad(y, (0, 32000 - len(y)))
    else:
        y = y[:32000]

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

    feat = np.array(log_mel_frames, dtype=np.float32)
    feat = (feat - g_mean) / g_std
    return feat, rms, max_amp

print("=" * 90)
print(" 🔬 AUDIT 10 SAMPEL UJIAN RESMI DI putar_sirene.py")
print("=" * 90)
print(f" {'No':<3} | {'Nama Ujian':<28} | {'Kelas Seharusnya':<16} | {'RMS Volume':<10} | {'Tebakan AI':<12} | {'Keyakinan':<10} | {'Status'}")
print("-" * 90)

for idx, (name, fpath, true_cls) in enumerate(EXAM_10_SAMPLES, 1):
    if not os.path.exists(fpath):
        print(f" {idx:2d} | {name:<28} | FILE TIDAK DITEMUKAN: {fpath}")
        continue
        
    feat, rms, max_amp = extract_feat(fpath)
    expected_shape = interp.get_input_details()[0]['shape']
    feat = np.reshape(feat, expected_shape)
    
    if is_quant:
        input_data = np.round(feat / in_scale + in_zero).astype(np.int8)
    else:
        input_data = feat
        
    interp.set_tensor(in_idx, input_data)
    interp.invoke()
    out = interp.get_tensor(out_idx)[0]
    
    if is_quant:
        probs = (out.astype(np.float32) - out_zero) * out_scale
    else:
        probs = out
        
    pred_idx = int(np.argmax(probs))
    pred_cls = CATEGORIES[pred_idx]
    conf = probs[pred_idx] * 100.0
    
    status = "✅ COCOK" if pred_cls == true_cls else "❌ SALAH TEBAK"
    if rms < 0.01:
        status += " (SUARA KECIL/HENING!)"
        
    print(f" {idx:2d} | {name:<28} | {true_cls:<16} | {rms:<10.4f} | {pred_cls:<12} | {conf:<8.1f}% | {status}")

print("=" * 90)
