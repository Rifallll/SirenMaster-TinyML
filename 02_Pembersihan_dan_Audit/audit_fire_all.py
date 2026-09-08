"""
audit_fire_all.py
=================
Mengaudit seluruh file DAMKAR (FIRETRUCK) di dalam folder FIRETRUCK
untuk memeriksa konsistensi deteksi antara Scaler Latihan (04_Training_AI)
dan Scaler ESP32 (model.h), serta mencari sampel DAMKAR yang 1000% kebal
dari salah tebak menjadi Ambulance.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, glob, librosa, numpy as np
import tensorflow as tf

ROOT = r"C:\Users\ASUS\Videos\DATASET"
TFLITE = os.path.join(ROOT, "sirenmaster_main", "model.tflite")
if not os.path.exists(TFLITE):
    TFLITE = os.path.join(ROOT, "04_Training_AI", "siren_model_quant.tflite")

SCALER_TRAIN = os.path.join(ROOT, "04_Training_AI", "siren_scaler.npz")
SCALER_ESP32 = os.path.join(ROOT, "siren_scaler.npz")

CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

interp = tf.lite.Interpreter(model_path=TFLITE)
interp.allocate_tensors()
in_idx = interp.get_input_details()[0]['index']
out_idx = interp.get_output_details()[0]['index']
in_scale, in_zero = interp.get_input_details()[0]['quantization']
out_scale, out_zero = interp.get_output_details()[0]['quantization']
is_quant = (in_scale != 0.0)
expected_shape = interp.get_input_details()[0]['shape']

# Load kedua scaler
scaler_train_data = np.load(SCALER_TRAIN)
g_mean_train, g_std_train = scaler_train_data['global_mean'], scaler_train_data['global_std']

scaler_esp32_data = np.load(SCALER_ESP32)
g_mean_esp32, g_std_esp32 = scaler_esp32_data['global_mean'], scaler_esp32_data['global_std']

def extract_mel_with_scaler(y, mean_arr, std_arr):
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
    feat = (feat - mean_arr) / std_arr
    return np.reshape(feat, expected_shape)

def predict_feat(feat):
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
    return CATEGORIES[pred_idx], probs[pred_idx] * 100.0, probs

print("=" * 85)
print(" 🚒 AUDIT SELURUH FILE DAMKAR (FIRETRUCK): PERBANDINGAN SCALER & DETEKSI")
print("=" * 85)

fire_folder = os.path.join(ROOT, "FIRETRUCK")
fire_files = sorted(glob.glob(os.path.join(fire_folder, "fire_*.wav")))

correct_train = 0
correct_esp32 = 0
total_tested = 0
super_pure_fire = []

for fp in fire_files:
    fn = os.path.basename(fp)
    try:
        y, _ = librosa.load(fp, sr=8000, duration=4.0)
    except Exception:
        continue
    rms = np.sqrt(np.mean(y**2))
    if rms < 0.20:
        continue
    total_tested += 1
    
    # Prediksi dengan scaler train
    feat_train = extract_mel_with_scaler(y, g_mean_train, g_std_train)
    cls_train, conf_train, _ = predict_feat(feat_train)
    if cls_train == "FIRETRUCK": correct_train += 1
    
    # Prediksi dengan scaler esp32
    feat_esp32 = extract_mel_with_scaler(y, g_mean_esp32, g_std_esp32)
    cls_esp32, conf_esp32, _ = predict_feat(feat_esp32)
    if cls_esp32 == "FIRETRUCK": correct_esp32 += 1
    
    # Cek apakah robust di kedua scaler dan pada volume rendah/tinggi
    if cls_train == "FIRETRUCK" and cls_esp32 == "FIRETRUCK" and conf_train >= 99.0 and conf_esp32 >= 99.0:
        # Cek volume g=0.5 dan g=1.5
        cls_low, conf_low, _ = predict_feat(extract_mel_with_scaler(y * 0.5, g_mean_esp32, g_std_esp32))
        cls_high, conf_high, _ = predict_feat(extract_mel_with_scaler(y * 1.5, g_mean_esp32, g_std_esp32))
        if cls_low == "FIRETRUCK" and cls_high == "FIRETRUCK" and conf_low >= 98.0 and conf_high >= 98.0:
            super_pure_fire.append((fn, rms, conf_esp32))

print(f"\n[*] Total sampel DAMKAR (fire_*.wav) yang diuji : {total_tested}")
print(f"[*] Akurasi dengan Scaler Training (04_Training_AI): {correct_train}/{total_tested} ({correct_train/total_tested*100:.1f}%)")
print(f"[*] Akurasi dengan Scaler ESP32 (`model.h` / Root) : {correct_esp32}/{total_tested} ({correct_esp32/total_tested*100:.1f}%)")

print("\n" + "=" * 85)
print(" 🏆 10 SAMPEL DAMKAR TERKUAT (100% ANTI-SALAH TEBAK JADI AMBULANCE DI ESP32 & PYTHON)")
print("=" * 85)

for fn, rms, c in sorted(super_pure_fire, key=lambda x: x[2], reverse=True)[:10]:
    print(f"    -> [SUPER FIRETRUCK] {fn:<32} | RMS: {rms:.4f} | Conf ESP32: {c:.1f}%")
