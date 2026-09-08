"""
audit_semua_kelas.py
====================
Audit menyeluruh: apakah model TFLite BENAR mendeteksi setiap kelas (Ambulance, Damkar, Polisi)
saat diuji dengan audio ASLI dari dataset dengan normalisasi yang 100% sama dengan ESP32.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, glob, numpy as np, librosa, tensorflow as tf

ROOT = r"C:\Users\ASUS\Videos\DATASET"

# Cari model tflite
TFLITE = os.path.join(ROOT, "sirenmaster_main", "model.tflite")
if not os.path.exists(TFLITE):
    TFLITE = os.path.join(ROOT, "04_Training_AI", "siren_model_quant.tflite")
print("Model path:", TFLITE)
print("Model exists:", os.path.exists(TFLITE))

# Cari scaler
SCALER = os.path.join(ROOT, "siren_scaler.npz")
print("Scaler path:", SCALER)
print("Scaler exists:", os.path.exists(SCALER))

scaler = np.load(SCALER)
g_mean = scaler['global_mean']
g_std  = scaler['global_std']
print(f"Scaler shape: mean={g_mean.shape} std={g_std.shape}")
print(f"Scaler values: mean[0]={g_mean[0]:.4f}, std[0]={g_std[0]:.4f}")

interp = tf.lite.Interpreter(model_path=TFLITE)
interp.allocate_tensors()
in_d  = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
in_s, in_z = in_d['quantization']
out_s, out_z = out_d['quantization']
is_quant = (in_s != 0.0)

print(f"\nModel input shape: {in_d['shape']}")
print(f"Input dtype: {in_d['dtype']}, scale={in_s}, zero_point={in_z}")
print(f"Is quantized: {is_quant}")

CATS = ['AMBULANCE', 'FIRETRUCK', 'NOISE', 'POLICE']

ham   = np.hamming(256)
mel_fb = librosa.filters.mel(sr=8000, n_fft=256, n_mels=40, fmin=0, fmax=4000)

def predict(filepath):
    y, _ = librosa.load(filepath, sr=8000)
    dc = np.mean(y)
    y = y - dc
    nf = (len(y) - 256) // 128 + 1
    frames = []
    for f in range(nf):
        fd = y[f*128 : f*128+256]
        if len(fd) < 256:
            fd = np.pad(fd, (0, 256-len(fd)))
        p = np.abs(np.fft.rfft(fd * ham, n=256)) ** 2
        mel_e = np.dot(mel_fb, p)
        frames.append(np.log(mel_e + 1e-9))
    feat = np.array(frames, dtype=np.float32)
    feat = (feat - g_mean) / g_std
    feat = feat.reshape(in_d['shape'])
    if is_quant:
        inp = np.round(feat / in_s + in_z).astype(np.int8)
    else:
        inp = feat
    interp.set_tensor(in_d['index'], inp)
    interp.invoke()
    raw = interp.get_tensor(out_d['index'])[0]
    if is_quant:
        scores = (raw.astype(np.float32) - out_z) * out_s
    else:
        scores = raw.astype(np.float32)
    return scores

print()
print("=" * 80)
print(" AUDIT KELAS: AMBULANCE, FIRETRUCK, POLICE (5 File per Kelas)")
print("=" * 80)

tests = [
    ("AMBULANCE", os.path.join(ROOT, "AMBULANCE"), "ambulance_*.wav", 0),
    ("FIRETRUCK", os.path.join(ROOT, "FIRETRUCK"), "fire_0002_seg*.wav", 1),
    ("POLICE",    os.path.join(ROOT, "POLISI"),    "police_*.wav", 3),
]

for class_name, folder, pattern, expected_idx in tests:
    files = sorted(glob.glob(os.path.join(folder, pattern)))[:5]
    print(f"\n[+] Kelas {class_name} ({len(files)} file):")
    print(f"    {'File':<30} | {'A%':>4} {'F%':>4} {'N%':>4} {'P%':>4} | Prediksi      | Status")
    print("    " + "-"*75)
    for fp in files:
        fn = os.path.basename(fp)
        try:
            scores = predict(fp)
            scores_pos = np.maximum(0, scores)
            s = scores_pos / (scores_pos.sum() + 1e-9)
            pred = int(np.argmax(s))
            status = "BENAR" if pred == expected_idx else "SALAH ❌"
            print(f"    {fn:<30} | {s[0]*100:4.0f} {s[1]*100:4.0f} {s[2]*100:4.0f} {s[3]*100:4.0f} | {CATS[pred]:<13} | {status}")
        except Exception as e:
            print(f"    {fn:<30} | ERROR: {e}")

# Khusus: cek apakah scaler cocok dengan model.h
print()
print("=" * 80)
print(" CEK SINKRONISASI: siren_scaler.npz vs model.h (MEL_MEAN/MEL_STD)")
print("=" * 80)

modelh_path = os.path.join(ROOT, "sirenmaster_main", "model.h")
if os.path.exists(modelh_path):
    content = open(modelh_path, 'r', encoding='utf-8', errors='replace').read()
    import re
    mean_vals = re.findall(r'MEL_MEAN\[.*?PROGMEM\s*=\s*\{([^}]+)\}', content, re.DOTALL)
    std_vals  = re.findall(r'MEL_STD\[.*?PROGMEM\s*=\s*\{([^}]+)\}', content, re.DOTALL)
    if mean_vals:
        modelh_mean = [float(x) for x in mean_vals[0].strip().split(',') if x.strip()]
        modelh_mean = np.array(modelh_mean[:5])
        print(f"model.h MEL_MEAN[:5]: {modelh_mean}")
        print(f"scaler  MEL_MEAN[:5]: {g_mean[:5]}")
        diff = np.abs(modelh_mean - g_mean[:5]).max()
        print(f"Max difference : {diff:.6f} -> {'COCOK' if diff < 0.01 else 'TIDAK COCOK !!!'}")
    else:
        print("MEL_MEAN tidak ditemukan di model.h, cek format file.")
else:
    print("model.h tidak ditemukan!")
