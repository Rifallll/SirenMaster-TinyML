"""
check_fire_speaker_robustness.py
==================================
Mencari sampel audio FIRETRUCK dan POLICE yang 100% KEBAL / ROBUST ketika diputar
melalui speaker laptop (yang biasanya kehilangan bass di bawah 300Hz dan mengalami
peningkatan frekuensi menengah) saat ditangkap oleh mikrofon ESP32.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, glob, librosa, numpy as np
import tensorflow as tf
from scipy.signal import butter, lfilter

ROOT = r"C:\Users\ASUS\Videos\DATASET"
TFLITE = os.path.join(ROOT, "sirenmaster_main", "model.tflite")
if not os.path.exists(TFLITE):
    TFLITE = os.path.join(ROOT, "04_Training_AI", "siren_model_quant.tflite")
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
expected_shape = interp.get_input_details()[0]['shape']

# Simulasi speaker laptop (High pass 250Hz agar meniru hilangnya bass speaker PC)
def simulate_laptop_speaker(y, sr=8000):
    b, a = butter(2, 250.0 / (0.5 * sr), btype='high')
    y_filt = lfilter(b, a, y)
    return y_filt

def predict_audio(y):
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
    return CATEGORIES[pred_idx], probs[pred_idx] * 100.0, probs

print("=" * 85)
print(" 🚒 AUDIT KETAHANAN SPEAKER LAPTOP UNTUK SAMPLES FIRETRUCK")
print("=" * 85)

# Cek kelima sampel fire_0002_seg0X
for i in range(1, 6):
    fp = os.path.join(ROOT, "FIRETRUCK", f"fire_0002_seg0{i}.wav")
    if not os.path.exists(fp): continue
    y, _ = librosa.load(fp, sr=8000, duration=4.0)
    y_lap = simulate_laptop_speaker(y)
    
    print(f"\n---> fire_0002_seg0{i}.wav <---")
    for g in [0.5, 1.0, 1.5, 2.0]:
        p_cls, p_conf, probs = predict_audio(y_lap * g)
        status = "✅ FIRETRUCK" if p_cls == "FIRETRUCK" else f"❌ SALAH -> {p_cls}"
        print(f"    Gain {g:3.1f}x (Simulasi Speaker Laptop) -> Prediksi: {p_cls:<10} ({p_conf:4.1f}%) | {status}")

print("\n" + "=" * 85)
print(" 💎 MENCARI SAMPEL FIRETRUCK & POLICE YANG 100% KEBAL SIMULASI SPEAKER LAPTOP")
print("=" * 85)

for cat, prefix in [('FIRETRUCK', 'fire_'), ('POLICE', 'police_')]:
    print(f"\n[*] Seleksi sampel super-kebal untuk {cat}...")
    folder = os.path.join(ROOT, cat)
    wavs = sorted(glob.glob(os.path.join(folder, f"{prefix}*.wav")) + glob.glob(os.path.join(folder, "*.wav")))
    best_list = []
    seen = set()
    for fp in wavs:
        fn = os.path.basename(fp)
        if fn in seen or fn.startswith(("SYNTH_", "dl_v3_", "aug_")):
            continue
        seen.add(fn)
        try:
            y, _ = librosa.load(fp, sr=8000, duration=4.0)
        except Exception:
            continue
        rms = np.sqrt(np.mean(y**2))
        if rms < 0.25: continue
        
        y_lap = simulate_laptop_speaker(y)
        ok = True
        min_conf = 100.0
        for g in [0.6, 1.0, 1.4]:
            p_cls, p_conf, _ = predict_audio(y_lap * g)
            if p_cls != cat or p_conf < 98.0:
                ok = False
                break
            if p_conf < min_conf:
                min_conf = p_conf
        if ok:
            best_list.append((fn, rms, min_conf))
            print(f"    -> [KEBAL SPEAKER] {fn:<32} | RMS: {rms:.4f} | Min Conf: {min_conf:.1f}%")
            if len(best_list) >= 6:
                break
