import os
import numpy as np
import tensorflow as tf
import librosa
import re
from tqdm import tqdm

DATASET_BASE = r"C:\Users\ASUS\Videos\DATASET"
MODEL_H = os.path.join(DATASET_BASE, "sirenmaster_main", "model.h")
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

# Load scaling stats
with open(MODEL_H, 'r') as f:
    content = f.read()
mean_match = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\}', content)
std_match  = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\}', content)
MEL_MEAN = np.array([float(x.strip().rstrip('f')) for x in mean_match.group(1).split(',')])
MEL_STD  = np.array([float(x.strip().rstrip('f')) for x in std_match.group(1).split(',')])

# Load Keras model
keras_model = tf.keras.models.load_model(os.path.join(DATASET_BASE, "siren_classifier_model.h5"))

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=os.path.join(DATASET_BASE, "siren_model_quant.tflite"))
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

def extract_features(fpath):
    y, _ = librosa.load(fpath, sr=8000, mono=True, duration=4.0)
    if len(y) < 32000:
        repeats = int(np.ceil(32000 / len(y)))
        y = np.tile(y, repeats)[:32000]
    else:
        y = y[:32000]
    y = np.convolve(y, [1/3, 1/3, 1/3], mode='same')
    
    n_frames = (32000 - 256) // 128 + 1
    hamming = np.hamming(256)
    mel_fb  = librosa.filters.mel(sr=8000, n_fft=256, n_mels=40, fmin=0, fmax=4000)
    
    log_mel_frames = []
    for i in range(n_frames):
        start = i * 128
        frame = y[start:start+256].copy()
        frame -= np.mean(frame)
        frame *= hamming
        fft   = np.fft.rfft(frame, n=256)
        power = np.abs(fft)**2
        mel   = mel_fb @ power
        log_mel_frames.append(np.log(mel + 1e-9))
    spec = np.array(log_mel_frames, dtype=np.float32)
    spec = (spec - MEL_MEAN) / (MEL_STD + 1e-8)
    return spec

def predict_keras(spec):
    inp = spec[np.newaxis, ..., np.newaxis]
    probs = keras_model.predict(inp, verbose=0)[0]
    return probs

def predict_tflite(spec):
    inp = spec[np.newaxis, ..., np.newaxis].astype(np.float32)
    # Dynamic Range model: float32 I/O — no manual quantization needed
    interpreter.set_tensor(inp_det['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(out_det['index'])[0].astype(np.float32)
    return out

# Test on 100 random files across all classes
np.random.seed(42)
all_test_files = []
for idx, cat in enumerate(CATEGORIES):
    folder = os.path.join(DATASET_BASE, cat)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.wav') and not f.startswith('aug_')]
    selected = np.random.choice(files, min(25, len(files)), replace=False)
    for f in selected:
        all_test_files.append((f, idx))

discrepancies = 0
correct_keras = 0
correct_tflite = 0

for fp, true_label in all_test_files:
    spec = extract_features(fp)
    p_k = predict_keras(spec)
    p_t = predict_tflite(spec)
    pred_k = np.argmax(p_k)
    pred_t = np.argmax(p_t)
    
    if pred_k == true_label:
        correct_keras += 1
    if pred_t == true_label:
        correct_tflite += 1
        
    if pred_k != pred_t:
        discrepancies += 1
        print(f"Discrepancy in {os.path.basename(fp)}:")
        print(f"  True Label  : {CATEGORIES[true_label]} ({true_label})")
        print(f"  Keras Pred  : {CATEGORIES[pred_k]} (conf: {p_k[pred_k]:.4f}) -> {list(np.round(p_k, 3))}")
        print(f"  TFLite Pred : {CATEGORIES[pred_t]} (conf: {p_t[pred_t]:.4f}) -> {list(np.round(p_t, 3))}")

print("\n" + "="*50)
print(f"Evaluated {len(all_test_files)} files.")
print(f"Keras Accuracy  : {correct_keras}/{len(all_test_files)} ({100*correct_keras/len(all_test_files):.1f}%)")
print(f"TFLite Accuracy : {correct_tflite}/{len(all_test_files)} ({100*correct_tflite/len(all_test_files):.1f}%)")
print(f"Discrepancies   : {discrepancies}/{len(all_test_files)}")
print("="*50)
