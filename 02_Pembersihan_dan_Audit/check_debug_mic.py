import os
import sys
import numpy as np
import librosa
import tensorflow as tf
import re

DATASET_BASE = r"C:\Users\ASUS\Videos\DATASET"
WAV_PATH = os.path.join(DATASET_BASE, "debug_mic_test.wav")
MODEL_PATH = os.path.join(DATASET_BASE, "siren_model_quant.tflite")
MODEL_H = os.path.join(DATASET_BASE, "sirenmaster_main", "model.h")
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

SAMPLE_RATE = 8000
DURATION = 4.0
TARGET_LEN = int(SAMPLE_RATE * DURATION)

if not os.path.exists(WAV_PATH):
    print(f"ERROR: {WAV_PATH} tidak ditemukan!")
    sys.exit(1)

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
    if inp_det['dtype'] == np.int8 or inp_det['dtype'] == np.uint8:
        sc = inp_det['quantization_parameters']['scales'][0]
        zp = inp_det['quantization_parameters']['zero_points'][0]
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

# Load audio
y, sr = librosa.load(WAV_PATH, sr=SAMPLE_RATE)
print(f"Loaded {WAV_PATH}: length={len(y)} samples ({len(y)/SAMPLE_RATE:.2f} seconds), sr={sr}")

# In offline mode, the audio in debug_mic_test.wav was saved NORMALIZED!
# Let's check how it predicts if we scale it down to original recorded level.
# The original max amplitude was about 0.010437 or 0.033813.
# The wav file was saved after normalization: y_normalized = y / max_amp.
# Let's evaluate the normalized version first (which is louder and cleaner).
print("\n--- EVALUATING NORMALIZED RECORDING ---")
spec = extract_features(y)
probs = predict(spec)
probs_adj = probs * np.array([0.80, 1.20, 1.00, 1.00])
probs_adj /= probs_adj.sum()
pred_idx = np.argmax(probs_adj)
print("Raw probabilities:")
for i, cat in enumerate(CATEGORIES):
    print(f"  {cat:12s}: {probs[i]*100:.1f}%")
print("Adjusted probabilities:")
for i, cat in enumerate(CATEGORIES):
    print(f"  {cat:12s}: {probs_adj[i]*100:.1f}%")
print(f"Prediction: {CATEGORIES[pred_idx]}")

# Let's scale it down to the original quiet level (e.g. max amplitude = 0.03)
print("\n--- EVALUATING ORIGINAL QUIET RECORDING (max_amp = 0.03) ---")
y_quiet = y * 0.03
spec_quiet = extract_features(y_quiet)
probs_quiet = predict(spec_quiet)
probs_quiet_adj = probs_quiet * np.array([0.80, 1.20, 1.00, 1.00])
probs_quiet_adj /= probs_quiet_adj.sum()
pred_quiet_idx = np.argmax(probs_quiet_adj)
print("Raw probabilities:")
for i, cat in enumerate(CATEGORIES):
    print(f"  {cat:12s}: {probs_quiet[i]*100:.1f}%")
print("Adjusted probabilities:")
for i, cat in enumerate(CATEGORIES):
    print(f"  {cat:12s}: {probs_quiet_adj[i]*100:.1f}%")
print(f"Prediction: {CATEGORIES[pred_quiet_idx]}")
