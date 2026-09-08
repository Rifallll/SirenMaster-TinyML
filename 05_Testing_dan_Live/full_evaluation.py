# full_evaluation.py – evaluate the INT8 TFLite model on the whole dataset
# -------------------------------------------------
# This script automatically:
#   1. Parses MFCC_MEAN and MFCC_STD from the generated model.h file.
#   2. Walks through each class folder (AMBULANCE, FIRETRUCK, NORMAL, POLICE).
#   3. Extracts MFCC features (same parameters as training).
#   4. Runs INT8 inference using TensorFlow Lite.
#   5. Computes confusion matrix, classification report and prints false‑positive examples.
# -------------------------------------------------
import os, re, json
import numpy as np
import librosa
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# ---------- CONFIG ----------
MODEL_PATH = "siren_model_quant.tflite"
MODEL_H_PATH = os.path.expanduser(r"C:\Users\ASUS\Documents\Arduino\sirenmaster_main\sirenmaster_main\model.h")
DATASET_ROOT = r"C:\Users\ASUS\Videos\DATASET"
SAMPLE_RATE = 8000
DURATION = 1.024               # 8192 samples
TARGET_LEN = int(SAMPLE_RATE * DURATION)
CLASSES = ["AMBULANCE", "FIRETRUCK", "NORMAL", "POLICE"]

# ---------- HELPERS ----------
def parse_array_from_header(header_path, name):
    """Parse a float array defined as:
       const float MFCC_MEAN[13] PROGMEM = { 0.1234f, ... };
       Returns a NumPy array of shape (13,).
    """
    with open(header_path, "r", encoding="utf-8") as f:
        txt = f.read()
    pattern = rf"const float {name}\[\d+\]\s+PROGMEM\s*=\s*{{([^}}]+)}};"
    m = re.search(pattern, txt, re.MULTILINE)
    if not m:
        raise ValueError(f"{name} not found in {header_path}")
    raw = m.group(1)
    # split by commas, strip trailing 'f' and whitespace
    values = [float(v.replace('f', '').strip()) for v in raw.split(',') if v.strip()]
    return np.array(values, dtype=np.float32)

def load_wav(fp):
    y, sr = librosa.load(fp, sr=SAMPLE_RATE)
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    else:
        y = y[:TARGET_LEN]
    return y.astype(np.float32)

def mfcc_feat(y):
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE,
        n_mfcc=13, n_fft=256,
        hop_length=128, n_mels=40,
        fmin=0, fmax=4000,
        window='hamming')
    return mfcc.T  # (63,13)

# ---------- MAIN ----------
print("Parsing MFCC mean/std from model.h …")
MFCC_MEAN = parse_array_from_header(MODEL_H_PATH, "MFCC_MEAN")
MFCC_STD  = parse_array_from_header(MODEL_H_PATH, "MFCC_STD")
print("Mean:", MFCC_MEAN)
print("Std :", MFCC_STD)

# Load TFLite interpreter (INT8)
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

true_labels = []
pred_labels = []
false_examples = []  # store (filepath, true, pred, confidence)

for idx, cls in enumerate(CLASSES):
    folder = os.path.join(DATASET_ROOT, cls)
    for fname in os.listdir(folder):
        if not fname.lower().endswith('.wav'):
            continue
        fp = os.path.join(folder, fname)
        y = load_wav(fp)
        feat = mfcc_feat(y)
        # Z‑score normalisation (same as training)
        feat = (feat - MFCC_MEAN) / MFCC_STD
        
        # Quantize to INT8 using model's input details
        scale, zero_point = input_details['quantization']
        if scale > 0.0:
            feat_q = np.round(feat / scale + zero_point)
            feat_q = np.clip(feat_q, -128, 127).astype(np.int8)
        else:
            feat_q = feat.astype(np.int8)
            
        inp = np.expand_dims(feat_q, axis=0)  # (1,63,13)
        interpreter.set_tensor(input_details['index'], inp)
        interpreter.invoke()
        probs_q = interpreter.get_tensor(output_details['index'])[0]
        
        # Dequantize output probabilities
        out_scale, out_zp = output_details['quantization']
        if out_scale > 0.0:
            probs = (probs_q.astype(np.float32) - out_zp) * out_scale
        else:
            probs = probs_q
            
        pred = int(np.argmax(probs))
        true_labels.append(idx)
        pred_labels.append(pred)
        if pred != idx:
            false_examples.append((fp, cls, CLASSES[pred], float(probs[pred])))

# Compute metrics
cm = confusion_matrix(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels, target_names=CLASSES, digits=4)

print("\n=== CONFUSION MATRIX ===")
print(cm)
print("\n=== CLASSIFICATION REPORT ===")
print(report)

# Show a few false‑positive examples per class
print("\n=== SAMPLE FALSE POSITIVES (max 5 per true class) ===")
per_class = {c: [] for c in CLASSES}
for fp, true, pred, conf in false_examples:
    if len(per_class[true]) < 5:
        per_class[true].append((fp, pred, conf))

for true_cls, items in per_class.items():
    if items:
        print(f"\n[True: {true_cls}]")
        for fp, pred, conf in items:
            print(f"  {fp} -> {pred} (conf={conf:.3f})")

# Save results to JSON for later analysis
result = {
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "false_examples": [
        {"filepath": fp, "true": true, "pred": pred, "confidence": conf}
        for fp, true, pred, conf in false_examples
    ]
}
with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print("\nResults saved to evaluation_results.json")
