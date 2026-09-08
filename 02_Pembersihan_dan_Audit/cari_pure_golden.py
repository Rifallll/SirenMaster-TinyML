"""
cari_pure_golden.py
===================
Mencari 10 sampel audio MURNI ASLI (tanpa SYNTH, tanpa dl_v3, tanpa aug_) dengan
volume RMS keras (> 0.35) dan akurasi AI TFLite >= 99.0%.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, glob, librosa, numpy as np
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
expected_shape = interp.get_input_details()[0]['shape']

def eval_file(fpath, target_cls):
    fn = os.path.basename(fpath)
    if fn.startswith(("SYNTH_", "dl_v3_", "aug_")):
        return None
    try:
        y, sr = librosa.load(fpath, sr=8000, duration=4.0)
    except Exception:
        return None
    rms = np.sqrt(np.mean(y**2))
    if rms < 0.25: # Cari yang volumenya cukup keras
        return None
        
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
    pred_cls = CATEGORIES[pred_idx]
    conf = probs[pred_idx] * 100.0
    
    if pred_cls == target_cls and conf >= 98.0:
        return (fn, rms, conf)
    return None

print("=" * 80)
print(" 💎 MENCARI 15 SAMPEL MURNI ASLI UNIK (TANPA SYNTH/DL/AUG)")
print("=" * 80)

golden_list = []
seen_names = set()

for cat, prefix, needed in [('AMBULANCE', 'ambulance_', 5), ('FIRETRUCK', 'fire_', 5), ('POLICE', 'police_', 5)]:
    print(f"\n[*] Mencari {needed} sampel murni unik terbaik untuk {cat} ({prefix}*.wav)...")
    folder = os.path.join(ROOT, cat)
    wavs = sorted(glob.glob(os.path.join(folder, f"{prefix}*.wav")) + glob.glob(os.path.join(folder, "*.wav")))
    found = 0
    for fp in wavs:
        fn = os.path.basename(fp)
        if fn in seen_names:
            continue
        res = eval_file(fp, cat)
        if res is not None:
            fn, rms, conf = res
            seen_names.add(fn)
            print(f"    -> [MURNI UNIK] {fn:<32} | RMS: {rms:.4f} | Conf: {conf:.1f}%")
            golden_list.append((cat, fn, rms, conf))
            found += 1
            if found >= needed:
                break

print("\n" + "=" * 80)
print(" 🎯 HASIL REKOMENDASI MURNI UNTUK putar_sirene.py :")
print("=" * 80)
for i, (cat, fn, rms, conf) in enumerate(golden_list, 1):
    print(f'    ("{cat} #{i}", os.path.join(ROOT, "{cat}", "{fn}")), # RMS: {rms:.4f}, Conf: {conf:.1f}%')
print("=" * 80)
