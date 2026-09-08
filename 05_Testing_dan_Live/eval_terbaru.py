"""
Evaluasi model terbaru dan temukan file-file yang masih salah klasifikasi
khususnya yang berkaitan dengan AMBULANCE
"""
import os, sys, json, numpy as np, librosa
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATASET_BASE = r"C:\Users\ASUS\Videos\DATASET"
MODEL_PATH   = os.path.join(DATASET_BASE, "siren_model_quant.tflite")
CATEGORIES   = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
SAMPLE_RATE  = 8000
N_FFT        = 256
HOP_LENGTH   = 128
N_MELS       = 40
DURATION     = 4.0
TARGET_LEN   = int(SAMPLE_RATE * DURATION)

# Load normalization stats dari model.h
import re
MODEL_H = os.path.join(DATASET_BASE, "siren_detection", "model.h")
if not os.path.exists(MODEL_H):
    MODEL_H = os.path.join(DATASET_BASE, "sirenmaster_main", "model.h")

with open(MODEL_H, 'r') as f:
    content = f.read()
mean_match = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\}', content)
std_match  = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\}', content)
MEL_MEAN = np.array([float(x.strip().rstrip('f')) for x in mean_match.group(1).split(',')])
MEL_STD  = np.array([float(x.strip().rstrip('f')) for x in std_match.group(1).split(',')])

_hamming = np.hamming(N_FFT).astype(np.float32)
_mel_fb  = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000)

def extract_features(y):
    if len(y) < TARGET_LEN:
        repeats = int(np.ceil(TARGET_LEN / len(y)))
        y = np.tile(y, repeats)[:TARGET_LEN]
    else:
        y = y[:TARGET_LEN]
        
    # NORMALIZE AUDIO (Auto-Gain) with max 10x boost
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        gain = min(1.0 / max_val, 10.0)
        y = y * gain
        
    y = np.convolve(y, [1/3, 1/3, 1/3], mode='same')
    n_frames = (TARGET_LEN - N_FFT) // HOP_LENGTH + 1
    log_mel_frames = []
    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = y[start:start+N_FFT].copy()
        frame -= np.mean(frame)
        frame *= _hamming
        fft   = np.fft.rfft(frame, n=N_FFT)
        power = np.abs(fft)**2
        mel   = _mel_fb @ power[:N_FFT//2+1]
        log_mel_frames.append(np.log(mel + 1e-9))
    spec = np.array(log_mel_frames, dtype=np.float32)  # (249, 40)
    spec = (spec - MEL_MEAN) / (MEL_STD + 1e-8)
    return spec

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

def predict(spec):
    inp = spec[np.newaxis, :, :, np.newaxis].astype(np.float32)
    if inp_det['dtype'] == np.int8 or inp_det['dtype'] == np.uint8:
        sc  = inp_det['quantization_parameters']['scales'][0]
        zp  = inp_det['quantization_parameters']['zero_points'][0]
        inp_q = np.clip(np.round(inp / sc) + zp, -128, 127).astype(np.int8)
        interpreter.set_tensor(inp_det['index'], inp_q)
        interpreter.invoke()
        out = interpreter.get_tensor(out_det['index'])[0].astype(np.float32)
        sc2 = out_det['quantization_parameters']['scales'][0]
        zp2 = out_det['quantization_parameters']['zero_points'][0]
        probs = (out - zp2) * sc2
    else:
        interpreter.set_tensor(inp_det['index'], inp)
        interpreter.invoke()
        probs = interpreter.get_tensor(out_det['index'])[0].astype(np.float32)
    return probs

print("=" * 60)
print("  EVALUASI MODEL BARU — DETEKSI FILE MASALAH")
print("=" * 60)

errors = []
counts = {c: {'total': 0, 'benar': 0} for c in CATEGORIES}

for cat_idx, cat in enumerate(CATEGORIES):
    folder = os.path.join(DATASET_BASE, cat)
    files  = [f for f in os.listdir(folder) if f.lower().endswith(('.wav', '.m4a'))]
    print(f"\n[{cat}] Mengevaluasi {len(files)} file...", flush=True)
    
    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            y, _ = librosa.load(fpath, sr=SAMPLE_RATE, mono=True, duration=DURATION)
            spec  = extract_features(y)
            probs = predict(spec)
            pred_idx  = int(np.argmax(probs))
            pred_conf = float(probs[pred_idx])
            counts[cat]['total'] += 1
            if pred_idx == cat_idx:
                counts[cat]['benar'] += 1
            else:
                errors.append({
                    'file': fname,
                    'folder': cat,
                    'true': cat,
                    'pred': CATEGORIES[pred_idx],
                    'conf': pred_conf,
                    'probs': probs.tolist()
                })
        except Exception as e:
            pass

print("\n" + "=" * 60)
print("  HASIL EVALUASI PER KATEGORI:")
print("=" * 60)
total_benar = 0
total_all   = 0
for cat in CATEGORIES:
    b = counts[cat]['benar']
    t = counts[cat]['total']
    total_benar += b
    total_all   += t
    pct = 100*b/t if t > 0 else 0
    print(f"  {cat:12s}: {b:4d}/{t:4d} benar ({pct:.1f}%)")

print(f"  {'TOTAL':12s}: {total_benar}/{total_all} ({100*total_benar/total_all:.1f}%)")

print("\n" + "=" * 60)
print("  FILE YANG MASIH SALAH (confidence >= 85%):")
print("=" * 60)

high_conf_errors = [e for e in errors if e['conf'] >= 0.85]
high_conf_errors.sort(key=lambda x: (-x['conf'], x['true'], x['pred']))

by_confusion = {}
for e in high_conf_errors:
    key = f"{e['true']} → {e['pred']}"
    by_confusion.setdefault(key, []).append(e)

for key, items in sorted(by_confusion.items()):
    print(f"\n  [{key}] — {len(items)} file:")
    for e in items[:10]:  # max 10 per grup
        print(f"    ❌ {e['file'][:55]:55s} ({e['conf']*100:.1f}%)")
    if len(items) > 10:
        print(f"    ... dan {len(items)-10} file lainnya")

# Simpan hasil
result = {
    'counts': counts,
    'errors_high_conf': high_conf_errors,
    'by_confusion': {k: [{'file': e['file'], 'folder': e['folder'], 'conf': e['conf']} for e in v] 
                     for k, v in by_confusion.items()}
}
with open('eval_terbaru.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n\n[OK] Total error tinggi keyakinan: {len(high_conf_errors)} file")
print(f"[OK] Hasil disimpan ke eval_terbaru.json")
