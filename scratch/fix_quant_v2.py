"""
fix_quant_v2.py
Coba 3 strategi quantization dan pilih yang terbaik:
  A. Float32 TFLite (no quant) — baseline, terbesar, paling akurat
  B. Dynamic Range (weight-only INT8) — size ~50%, accuracy ~sama
  C. Float16 quantization — size ~50%, good GPU/CPU compatibility
  D. Full INT8 (sebelumnya) — size terkecil tapi sering jelek di BatchNorm model

Pilih yang accuracy gap < 2% dengan size terkecil.
"""
import os, sys
import numpy as np
import tensorflow as tf
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATASET_BASE = r"C:\Users\ASUS\Videos\DATASET"
CACHE_PATH   = os.path.join(DATASET_BASE, "siren_40x249_melspec_cache_lokal.npz")
SCALER_PATH  = os.path.join(DATASET_BASE, "siren_scaler.npz")
MODEL_H5     = os.path.join(DATASET_BASE, "siren_classifier_model.h5")
OUT_TFLITE   = os.path.join(DATASET_BASE, "siren_model_quant.tflite")
OUT_HEADER   = os.path.join(DATASET_BASE, "sirenmaster_main", "model.h")
CATEGORIES   = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

SAMPLE_RATE = 8000
N_FFT       = 256
HOP_LENGTH  = 128
N_MELS      = 40
N_MFCC      = 13
DURATION    = 4.0

print("=" * 65)
print("  TFLite Quantization Strategy Comparison v2")
print("=" * 65)

# Load scaler
scaler_data = np.load(SCALER_PATH)
global_mean = scaler_data['global_mean']
global_std  = scaler_data['global_std']

# Load cache
print("\n[1] Loading feature cache...")
with np.load(CACHE_PATH, allow_pickle=True) as data:
    X_train = data['X_train']
    y_train = data['y_train']
    X_test  = data['X_test']
    y_test  = data['y_test']
print(f"    X_train: {X_train.shape}, X_test: {X_test.shape}")

# Scale
X_train_s = ((X_train - global_mean[np.newaxis,np.newaxis,:]) /
             (global_std[np.newaxis,np.newaxis,:] + 1e-8))[..., np.newaxis].astype(np.float32)
X_test_s  = ((X_test  - global_mean[np.newaxis,np.newaxis,:]) /
             (global_std[np.newaxis,np.newaxis,:] + 1e-8))[..., np.newaxis].astype(np.float32)

# Load model
print("\n[2] Loading Keras model...")
model = tf.keras.models.load_model(MODEL_H5)

# Keras accuracy
print("\n[3] Keras baseline accuracy...")
y_pred_k = np.argmax(model.predict(X_test_s, batch_size=256, verbose=0), axis=1)
keras_acc = np.mean(y_pred_k == y_test)
print(f"    Keras accuracy: {keras_acc*100:.2f}%")

def eval_tflite(tflite_bytes):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    preds = []
    for i in range(len(X_test_s)):
        sample = X_test_s[i:i+1]
        if inp['dtype'] == np.int8:
            sc = inp['quantization_parameters']['scales'][0]
            zp = inp['quantization_parameters']['zero_points'][0]
            sample = np.clip(np.round(sample / sc) + zp, -128, 127).astype(np.int8)
        interp.set_tensor(inp['index'], sample)
        interp.invoke()
        raw = interp.get_tensor(out['index'])[0]
        if out['dtype'] == np.int8:
            sc2 = out['quantization_parameters']['scales'][0]
            zp2 = out['quantization_parameters']['zero_points'][0]
            raw = (raw.astype(np.float32) - zp2) * sc2
        preds.append(int(np.argmax(raw)))
        if (i+1) % 500 == 0:
            print(f"      ... {i+1}/{len(X_test_s)}", flush=True)
    acc = np.mean(np.array(preds) == y_test)
    return acc, np.array(preds)

results = {}

# ── Strategy A: Float32 (no quantization) ───────────────────────────
print("\n[A] Float32 TFLite (no quantization)...")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
m = conv.convert()
acc_a, _ = eval_tflite(m)
results['A_float32'] = {'bytes': len(m), 'acc': acc_a}
print(f"    Size: {len(m)/1024:.1f} KB  Accuracy: {acc_a*100:.2f}%  Drop: {(keras_acc-acc_a)*100:.2f}%")

# ── Strategy B: Dynamic Range (weight-only INT8) ─────────────────────
print("\n[B] Dynamic Range Quantization (weight-only INT8)...")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
# No representative_dataset → only weights quantized
m = conv.convert()
acc_b, _ = eval_tflite(m)
results['B_dynamic'] = {'bytes': len(m), 'acc': acc_b}
print(f"    Size: {len(m)/1024:.1f} KB  Accuracy: {acc_b*100:.2f}%  Drop: {(keras_acc-acc_b)*100:.2f}%")

# ── Strategy C: Float16 quantization ────────────────────────────────
print("\n[C] Float16 Quantization...")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
m = conv.convert()
acc_c, _ = eval_tflite(m)
results['C_float16'] = {'bytes': len(m), 'acc': acc_c}
print(f"    Size: {len(m)/1024:.1f} KB  Accuracy: {acc_c*100:.2f}%  Drop: {(keras_acc-acc_c)*100:.2f}%")

# ── Choose best ──────────────────────────────────────────────────────
print("\n" + "="*65)
print("  COMPARISON RESULTS")
print("="*65)
print(f"  Keras baseline : {keras_acc*100:.2f}%")
print(f"  A Float32      : {results['A_float32']['acc']*100:.2f}%  ({results['A_float32']['bytes']/1024:.1f} KB)")
print(f"  B Dynamic INT8 : {results['B_dynamic']['acc']*100:.2f}%  ({results['B_dynamic']['bytes']/1024:.1f} KB)")
print(f"  C Float16      : {results['C_float16']['acc']*100:.2f}%  ({results['C_float16']['bytes']/1024:.1f} KB)")

# Pick best: smallest that has <2% drop
candidates = [
    ('B_dynamic', results['B_dynamic']),
    ('C_float16', results['C_float16']),
    ('A_float32', results['A_float32']),
]
chosen_key = None
chosen_m   = None
for key, info in candidates:
    drop = keras_acc - info['acc']
    if drop < 0.02:
        chosen_key = key
        print(f"\n  [CHOSEN] {key} — drop {drop*100:.2f}% < 2%, size {info['bytes']/1024:.1f} KB")
        break

if chosen_key is None:
    # Fallback: pick the one with least drop
    chosen_key = min(results.keys(), key=lambda k: keras_acc - results[k]['acc'])
    print(f"\n  [FALLBACK] No strategy < 2% drop. Choosing {chosen_key} (min drop).")

# Re-convert chosen
print(f"\n[4] Re-converting chosen strategy: {chosen_key}...")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
if chosen_key == 'B_dynamic':
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
elif chosen_key == 'C_float16':
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
# else: float32, no settings needed
chosen_m = conv.convert()

with open(OUT_TFLITE, 'wb') as f:
    f.write(chosen_m)
print(f"    Saved: {OUT_TFLITE} ({len(chosen_m)/1024:.1f} KB)")

# Final evaluation
acc_final, preds_final = eval_tflite(chosen_m)
print(f"\n[5] Final TFLite accuracy: {acc_final*100:.2f}%  (drop: {(keras_acc-acc_final)*100:.2f}%)")

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, preds_final))
print(classification_report(y_test, preds_final, target_names=CATEGORIES))

# Generate model.h
print(f"\n[6] Generating model.h header...")
import librosa
hamming  = np.hamming(N_FFT).astype(np.float32)
mel_fb   = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000).T
n_frames = (int(SAMPLE_RATE * DURATION) - N_FFT) // HOP_LENGTH + 1

with open(OUT_HEADER, 'w') as f:
    f.write("/*\n")
    f.write(" * model.h - Siren Classifier Deployment Header\n")
    f.write(" * Generated automatically by fix_quant_v2.py\n")
    f.write(f" * Strategy: {chosen_key}  Size: {len(chosen_m)} bytes ({len(chosen_m)/1024:.2f} KB)\n")
    f.write(f" * Keras acc: {keras_acc*100:.2f}%  TFLite acc: {acc_final*100:.2f}%\n")
    f.write(" */\n\n")
    f.write("#ifndef MODEL_H\n#define MODEL_H\n\n")
    f.write("#include <pgmspace.h>\n\n")
    f.write(f"#define SAMPLE_RATE_HZ   {SAMPLE_RATE}\n")
    f.write(f"#define N_FFT_SIZE       {N_FFT}\n")
    f.write(f"#define N_HOP_LENGTH     {HOP_LENGTH}\n")
    f.write(f"#define N_MFCC_COEFF     {N_MFCC}\n")
    f.write(f"#define N_MEL_FILTERS    {N_MELS}\n")
    f.write(f"#define N_TIME_FRAMES    {n_frames}\n")
    f.write(f"#define N_FFT_BINS       {N_FFT // 2 + 1}\n")
    f.write(f"#define AUDIO_SAMPLES    {int(SAMPLE_RATE * DURATION)}\n")
    f.write(f"#define NUM_CLASSES      {len(CATEGORIES)}\n\n")

    f.write(f"const float HAMMING_WINDOW[{N_FFT}] PROGMEM = {{\n")
    for i, val in enumerate(hamming):
        if i % 8 == 0: f.write("  ")
        f.write(f"{val:.8f}f")
        if i < N_FFT - 1: f.write(", ")
        if (i+1) % 8 == 0 or i == N_FFT-1: f.write("\n")
    f.write("};\n\n")

    f.write(f"const float MEL_FILTERBANK[{N_MELS}][{N_FFT//2+1}] PROGMEM = {{\n")
    for m_i in range(N_MELS):
        f.write("  {")
        f.write(", ".join([f"{v:.8f}f" for v in mel_fb[:, m_i]]))
        f.write("}" + (",\n" if m_i < N_MELS-1 else "\n"))
    f.write("};\n\n")

    f.write(f"const float MEL_MEAN[{N_MELS}] PROGMEM = {{\n  ")
    f.write(", ".join([f"{v:.8f}f" for v in global_mean]))
    f.write("\n};\n\n")

    f.write(f"const float MEL_STD[{N_MELS}] PROGMEM = {{\n  ")
    f.write(", ".join([f"{v:.8f}f" for v in global_std]))
    f.write("\n};\n\n")

    f.write("#ifdef __has_attribute\n#define MODEL_ALIGN __attribute__((aligned(4)))\n")
    f.write("#else\n#define MODEL_ALIGN\n#endif\n\n")
    f.write(f"const unsigned int siren_model_data_len = {len(chosen_m)};\n\n")
    f.write("const unsigned char siren_model_data[] MODEL_ALIGN = {\n")
    for i, val in enumerate(chosen_m):
        if i % 12 == 0: f.write("  ")
        f.write(f"0x{val:02x}")
        if i < len(chosen_m)-1: f.write(", ")
        if (i+1) % 12 == 0 or i == len(chosen_m)-1: f.write("\n")
    f.write("};\n\n#endif // MODEL_H\n")

print(f"    model.h saved: {os.path.getsize(OUT_HEADER)//1024} KB")
print("\n[SUCCESS] Done!")
print(f"  Best strategy : {chosen_key}")
print(f"  TFLite size   : {len(chosen_m)/1024:.1f} KB")
print(f"  TFLite acc    : {acc_final*100:.2f}%  (Keras: {keras_acc*100:.2f}%)")
