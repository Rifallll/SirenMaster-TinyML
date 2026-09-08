"""
check_scaler_sync.py - Cek sinkronisasi siren_scaler.npz vs model.h + audit semua kelas
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import re, os, glob, numpy as np, librosa, tensorflow as tf

ROOT = r"C:\Users\ASUS\Videos\DATASET"
TFLITE = os.path.join(ROOT, "sirenmaster_main", "model.tflite")
if not os.path.exists(TFLITE):
    TFLITE = os.path.join(ROOT, "04_Training_AI", "siren_model_quant.tflite")

scaler = np.load(os.path.join(ROOT, "siren_scaler.npz"))
g_mean = scaler['global_mean']
g_std  = scaler['global_std']

# --- Cek model.h ---
modelh_path = os.path.join(ROOT, "sirenmaster_main", "model.h")
content = open(modelh_path, 'r', encoding='utf-8', errors='replace').read()
mean_match = re.findall(r'MEL_MEAN\[.*?PROGMEM\s*=\s*\{([^}]+)\}', content, re.DOTALL)
std_match  = re.findall(r'MEL_STD\[.*?PROGMEM\s*=\s*\{([^}]+)\}',  content, re.DOTALL)

print("=" * 65)
print(" CEK SINKRONISASI model.h vs siren_scaler.npz")
print("=" * 65)
if mean_match:
    raw_vals = [v.strip().rstrip('f') for v in mean_match[0].strip().split(',') if v.strip()]
    modelh_mean = np.array([float(v) for v in raw_vals if v])
    diff_mean = float(np.abs(modelh_mean - g_mean).max())
    status_mean = "COCOK ✅" if diff_mean < 0.001 else "TIDAK COCOK ❌"
    print(f"scaler mean[:3] = {g_mean[:3]}")
    print(f"modelh mean[:3] = {modelh_mean[:3]}")
    print(f"Max diff MEAN   = {diff_mean:.8f}  -> {status_mean}")
else:
    print("MEL_MEAN tidak ditemukan di model.h!")

if std_match:
    raw_vals = [v.strip().rstrip('f') for v in std_match[0].strip().split(',') if v.strip()]
    modelh_std = np.array([float(v) for v in raw_vals if v])
    diff_std = float(np.abs(modelh_std - g_std).max())
    status_std = "COCOK ✅" if diff_std < 0.001 else "TIDAK COCOK ❌"
    print(f"scaler std[:3]  = {g_std[:3]}")
    print(f"modelh std[:3]  = {modelh_std[:3]}")
    print(f"Max diff STD    = {diff_std:.8f}  -> {status_std}")
else:
    print("MEL_STD tidak ditemukan di model.h!")

# --- Audit prediksi semua kelas ---
interp = tf.lite.Interpreter(model_path=TFLITE)
interp.allocate_tensors()
in_d  = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
in_s, in_z   = in_d['quantization']
out_s, out_z = out_d['quantization']
is_quant = (in_s != 0.0)

CATS = ['AMBULANCE', 'FIRETRUCK', 'NOISE', 'POLICE']
ham    = np.hamming(256)
mel_fb = librosa.filters.mel(sr=8000, n_fft=256, n_mels=40, fmin=0, fmax=4000)

def predict(fp):
    y, _ = librosa.load(fp, sr=8000)
    y = y - np.mean(y)
    nf = (len(y) - 256) // 128 + 1
    frames = []
    for f in range(nf):
        fd = y[f*128:f*128+256]
        if len(fd) < 256: fd = np.pad(fd, (0, 256-len(fd)))
        mel_e = np.dot(mel_fb, np.abs(np.fft.rfft(fd*ham, n=256))**2)
        frames.append(np.log(mel_e + 1e-9))
    feat = (np.array(frames, np.float32) - g_mean) / g_std
    feat = feat.reshape(in_d['shape'])
    inp = np.round(feat/in_s+in_z).astype(np.int8) if is_quant else feat
    interp.set_tensor(in_d['index'], inp)
    interp.invoke()
    raw = interp.get_tensor(out_d['index'])[0]
    scores = (raw.astype(np.float32)-out_z)*out_s if is_quant else raw.astype(np.float32)
    return scores

print()
print("=" * 75)
print(" AUDIT PREDIKSI: AMBULANCE, FIRETRUCK, POLICE (5 file per kelas)")
print("=" * 75)

tests = [
    ("AMBULANCE", os.path.join(ROOT, "AMBULANCE"), "ambulance_*.wav",    0),
    ("FIRETRUCK", os.path.join(ROOT, "FIRETRUCK"), "fire_0002_seg*.wav", 1),
    ("POLICE",    os.path.join(ROOT, "POLICE"),    "police_*.wav",       3),
]

total_benar = 0
total_salah = 0
for class_name, folder, pattern, expected_idx in tests:
    files = sorted(glob.glob(os.path.join(folder, pattern)))[:5]
    if not files:
        print(f"\n[!] KELAS {class_name}: Folder tidak ada atau kosong -> {folder}")
        continue
    benar = 0
    print(f"\n[+] Kelas: {class_name} (folder: {folder})")
    print(f"    {'File':<35} |  A%  F%  N%  P% | Prediksi      | Status")
    print("    " + "-"*75)
    for fp in files:
        fn = os.path.basename(fp)
        try:
            s = predict(fp)
            s = np.maximum(0, s); s /= (s.sum()+1e-9)
            pred = int(np.argmax(s))
            ok = (pred == expected_idx)
            benar += int(ok)
            total_benar += int(ok)
            total_salah += int(not ok)
            status = "BENAR ✅" if ok else "SALAH ❌"
            print(f"    {fn:<35} | {s[0]*100:3.0f} {s[1]*100:3.0f} {s[2]*100:3.0f} {s[3]*100:3.0f} | {CATS[pred]:<13} | {status}")
        except Exception as e:
            print(f"    {fn:<35} | ERROR: {e}")
    print(f"    -> Akurasi kelas {class_name}: {benar}/{len(files)}")

print()
print(f"TOTAL BENAR: {total_benar} | TOTAL SALAH: {total_salah}")
print("=" * 75)
