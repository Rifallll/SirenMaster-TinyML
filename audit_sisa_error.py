import os
import sys
import numpy as np
import librosa
import tensorflow as tf
import re
from collections import Counter

DATASET_BASE = r"C:\Users\ASUS\Videos\DATASET"
MODEL_PATH = os.path.join(DATASET_BASE, "siren_model_quant.tflite")
MODEL_H = os.path.join(DATASET_BASE, "sirenmaster_main", "model.h")
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

SAMPLE_RATE = 8000
DURATION = 4.0
TARGET_LEN = int(SAMPLE_RATE * DURATION)

# Load stats dari model.h
with open(MODEL_H, 'r') as f:
    content = f.read()
mean_match = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\}', content)
std_match = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\}', content)
MEL_MEAN = np.array([float(x.strip().rstrip('f')) for x in mean_match.group(1).split(',')])
MEL_STD = np.array([float(x.strip().rstrip('f')) for x in std_match.group(1).split(',')])

_hamming = np.hamming(256).astype(np.float32)
_mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=256, n_mels=40, fmin=0, fmax=4000)

def extract_features(y):
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    else:
        y = y[:TARGET_LEN]
    y = np.convolve(y, [1/3, 1/3, 1/3], mode='same')
    n_frames = (TARGET_LEN - 256) // 128 + 1
    log_mel_frames = []
    for i in range(n_frames):
        start = i * 128
        frame = y[start:start+256].copy()
        frame -= np.mean(frame)
        frame *= _hamming
        fft = np.fft.rfft(frame, n=256)
        power = np.abs(fft)**2
        mel = _mel_fb @ power[:129]
        log_mel_frames.append(np.log(mel + 1e-9))
    spec = np.array(log_mel_frames, dtype=np.float32)
    spec = (spec - MEL_MEAN) / (MEL_STD + 1e-8)
    return spec

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

def predict(spec):
    inp = spec[np.newaxis, :, :, np.newaxis].astype(np.float32)
    sc = inp_det['quantization_parameters']['scales'][0]
    zp = inp_det['quantization_parameters']['zero_points'][0]
    inp_q = np.clip(np.round(inp / sc) + zp, -128, 127).astype(np.int8)
    interpreter.set_tensor(inp_det['index'], inp_q)
    interpreter.invoke()
    out = interpreter.get_tensor(out_det['index'])[0].astype(np.float32)
    sc2 = out_det['quantization_parameters']['scales'][0]
    zp2 = out_det['quantization_parameters']['zero_points'][0]
    probs = (out - zp2) * sc2
    return probs

def get_group_name(filename):
    base = os.path.splitext(filename)[0]
    # Remove aug prefixes
    base = re.sub(r'^(aug_noise_\d+|aug_pitch_\d+|aug_stretch_\d+)_', '', base)
    # Match root name
    match = re.match(r'^([a-zA-Z]+_\d+)', base)
    if match:
        return match.group(1)
    cleaned = base
    cleaned = re.sub(r'_(original|loud|noise_light|shift|seg\d+|_seg\d+)$', '', cleaned)
    return cleaned

high_conf_errors = []

for cat_idx, cat in enumerate(CATEGORIES):
    folder = os.path.join(DATASET_BASE, cat)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.wav', '.m4a'))]
    
    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            y, sr = librosa.load(fpath, sr=SAMPLE_RATE)
            chunks = []
            hop = TARGET_LEN
            
            for start in range(0, len(y) - TARGET_LEN + 1, hop):
                chunk = y[start:start+TARGET_LEN]
                rms = np.sqrt(np.mean(chunk**2))
                if rms >= 0.001:
                    chunks.append(chunk)
                    
            if len(y) < TARGET_LEN and len(y) >= TARGET_LEN // 2:
                rms = np.sqrt(np.mean(y**2))
                if rms >= 0.001:
                    padded = np.pad(y, (0, TARGET_LEN - len(y)), mode='constant')
                    chunks.append(padded)
                    
            for chunk in chunks:
                spec = extract_features(chunk)
                probs = predict(spec)
                probs_adj = probs * np.array([0.80, 1.20, 1.00, 1.00])
                probs_adj /= probs_adj.sum()
                
                pred_idx = int(np.argmax(probs_adj))
                conf = float(probs_adj[pred_idx])
                
                if pred_idx != cat_idx and conf >= 0.85:
                    high_conf_errors.append({
                        'file': fname,
                        'true': cat,
                        'pred': CATEGORIES[pred_idx],
                        'conf': conf,
                        'group': get_group_name(fname)
                    })
        except Exception as e:
            pass

print("=" * 60)
print("  AUDIT SISA ERROR KEYAKINAN TINGGI (>= 85%)")
print("=" * 60)

# Group errors
grouped = {}
for err in high_conf_errors:
    key = f"{err['true']} -> {err['pred']}"
    grouped.setdefault(key, []).append(err)

for key, items in sorted(grouped.items()):
    print(f"\n[{key}] — Total {len(items)} file:")
    # Group by source group
    group_counts = Counter([x['group'] for x in items])
    for g, count in group_counts.most_common(5):
        print(f"  - Grup '{g}': {count} file salah")
        # Print a sample filename
        sample_file = next(x['file'] for x in items if x['group'] == g)
        print(f"    (Contoh: {sample_file})")
print("=" * 60)
