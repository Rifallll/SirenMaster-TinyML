"""
check_model_robustness.py
===========================
Mengecek sensitivitas dan akurasi model AI pada berbagai level gain/volume
serta memindai file-file mana dari setiap kelas yang PALING ROBUST (kebal salah tebak)
saat diputar melalui speaker laptop ke mikrofon ESP32.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, glob, librosa, numpy as np
import tensorflow as tf

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
print(" 🔬 CEK KETAHANAN MODEL TERHADAP SAMPEL YANG DILAPORKAN USER")
print("=" * 85)

test_files = [
    ("FIRETRUCK (fire_0002_seg01.wav)", os.path.join(ROOT, "FIRETRUCK", "fire_0002_seg01.wav"), "FIRETRUCK"),
    ("FIRETRUCK (fire_0002_seg03.wav)", os.path.join(ROOT, "FIRETRUCK", "fire_0002_seg03.wav"), "FIRETRUCK"),
    ("FIRETRUCK (fire_0002_seg04.wav)", os.path.join(ROOT, "FIRETRUCK", "fire_0002_seg04.wav"), "FIRETRUCK"),
    ("POLICE (police_0001_seg05.wav)", os.path.join(ROOT, "POLICE", "police_0001_seg05.wav"), "POLICE"),
]

for name, fp, true_cls in test_files:
    if not os.path.exists(fp):
        continue
    y_raw, _ = librosa.load(fp, sr=8000, duration=4.0)
    print(f"\n---> {name} (Kelas Asli: {true_cls}) <---")
    for gain in [0.5, 1.0, 1.5, 3.0]:
        y_gained = y_raw * gain
        pred_cls, conf, probs = predict_audio(y_gained)
        probs_str = ", ".join([f"{CATEGORIES[i]}:{probs[i]*100:.0f}%" for i in range(4)])
        status = "✅ BENAR" if pred_cls == true_cls else f"❌ SALAH (Jadi {pred_cls})"
        print(f"    Gain {gain:3.1f}x (RMS {np.sqrt(np.mean(y_gained**2)):.3f}) -> Prediksi: {pred_cls:<10} ({conf:4.1f}%) | {status} | [{probs_str}]")

print("\n" + "=" * 85)
print(" 🏆 MENCARI 5 SAMPEL TERKUAT & PALING LANTANG PER KELAS (ANTI-SALAH TEBAK DI SPEAKER LAPTOP)")
print("=" * 85)

for cat in ['AMBULANCE', 'FIRETRUCK', 'POLICE']:
    print(f"\n[*] Seleksi sampel terkuat untuk {cat}...")
    folder = os.path.join(ROOT, cat)
    wavs = sorted(glob.glob(os.path.join(folder, "*.wav")))
    best_samples = []
    for fp in wavs:
        fn = os.path.basename(fp)
        if fn.startswith(("SYNTH_", "dl_v3_", "aug_")):
            continue
        try:
            y, _ = librosa.load(fp, sr=8000, duration=4.0)
        except Exception:
            continue
        rms = np.sqrt(np.mean(y**2))
        if rms < 0.35: # Hanya ambil yang super lantang (> 0.35)
            continue
            
        # Cek apakah dia tetap akurat > 99% baik pada gain 0.5x, 1.0x, maupun 2.0x
        ok = True
        min_conf = 100.0
        for g in [0.5, 1.0, 2.0]:
            p_cls, p_conf, _ = predict_audio(y * g)
            if p_cls != cat or p_conf < 98.0:
                ok = False
                break
            if p_conf < min_conf:
                min_conf = p_conf
        if ok:
            best_samples.append((fn, rms, min_conf))
            if len(best_samples) >= 5:
                break
                
    for fn, rms, c in best_samples:
        print(f"    -> [SUPER ROBUST] {fn:<32} | RMS: {rms:.4f} | Min Conf (0.5x-2x): {c:.1f}%")
