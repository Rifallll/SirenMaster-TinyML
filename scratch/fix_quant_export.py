"""
fix_quant_export.py
Re-export TFLite model with proper Full Integer Quantization.

Masalah sebelumnya:
- Hanya 300 samples sebagai representative dataset → kalibrasi quantization kurang presisi
- Input/output bertipe INT8 → susah diuji dan sering mismatch

Solusi:
- Gunakan lebih banyak representative samples (1000+) dari cache
- Gunakan FLOAT input/INT8 internal (lebih kompatibel dengan ESP32 inference)
- Validasi ulang TFLite vs Keras accuracy
"""
import os
import sys
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
SAMPLE_RATE  = 8000
N_FFT        = 256
HOP_LENGTH   = 128
N_MELS       = 40
N_MFCC       = 13
DURATION     = 4.0

print("=" * 65)
print("  TFLite Re-Export: Improved Full INT8 Quantization")
print("=" * 65)

# Load scaler
print("\n[1] Loading scaler parameters...")
scaler_data  = np.load(SCALER_PATH)
global_mean  = scaler_data['global_mean']
global_std   = scaler_data['global_std']
print(f"    Scaler loaded. Mean shape: {global_mean.shape}")

# Load feature cache for representative dataset
print("\n[2] Loading feature cache for representative dataset...")
with np.load(CACHE_PATH, allow_pickle=True) as data:
    X_train = data['X_train']
    y_train = data['y_train']
    X_test  = data['X_test']
    y_test  = data['y_test']
print(f"    Loaded X_train: {X_train.shape}, X_test: {X_test.shape}")

# Scale the data
X_train_scaled = (X_train - global_mean[np.newaxis, np.newaxis, :]) / (global_std[np.newaxis, np.newaxis, :] + 1e-8)
X_test_scaled  = (X_test  - global_mean[np.newaxis, np.newaxis, :]) / (global_std[np.newaxis, np.newaxis, :] + 1e-8)

# Add channel dim
X_train_scaled = X_train_scaled[..., np.newaxis].astype(np.float32)
X_test_scaled  = X_test_scaled[..., np.newaxis].astype(np.float32)
print(f"    Scaled & shaped: {X_train_scaled.shape}")

# Load Keras model
print("\n[3] Loading Keras model...")
model = tf.keras.models.load_model(MODEL_H5)
model.summary()

# Baseline accuracy
print("\n[4] Evaluating Keras baseline accuracy on test set...")
y_pred_keras = np.argmax(model.predict(X_test_scaled, batch_size=256, verbose=0), axis=1)
keras_acc = np.mean(y_pred_keras == y_test)
print(f"    Keras accuracy: {keras_acc*100:.2f}%")

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred_keras))
print(classification_report(y_test, y_pred_keras, target_names=CATEGORIES))

# Representative dataset generator (use stratified 1500 samples)
print("\n[5] Preparing representative dataset (1500 stratified samples)...")
rep_indices = []
for cls in range(len(CATEGORIES)):
    cls_idx = np.where(y_train == cls)[0]
    n = min(375, len(cls_idx))  # 375 per class = 1500 total
    rep_indices.extend(np.random.choice(cls_idx, n, replace=False).tolist())
np.random.shuffle(rep_indices)
X_rep = X_train_scaled[rep_indices]
print(f"    Representative dataset: {X_rep.shape} samples")

def representative_data_gen():
    for i in range(len(X_rep)):
        yield [X_rep[i:i+1]]

# Convert with FLOAT input / INT8 internal weights (hybrid quantization)
# This mode is MORE COMPATIBLE with ESP32 and avoids INT8 I/O precision loss
print("\n[6] Converting to TFLite with Full INT8 (float I/O for compatibility)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Keep input/output as float32 for easier inference on ESP32 + Python testing
# INT8 weights internally for size reduction
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
# DO NOT set inference_input_type / inference_output_type → stays float32 I/O
tflite_model = converter.convert()
print(f"    Converted! Size: {len(tflite_model)/1024:.2f} KB")

# Save
with open(OUT_TFLITE, 'wb') as f:
    f.write(tflite_model)
print(f"    Saved to: {OUT_TFLITE}")

# Validate TFLite accuracy
print("\n[7] Evaluating TFLite accuracy on test set...")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]
print(f"    TFLite input  type: {inp_det['dtype']}, shape: {inp_det['shape']}")
print(f"    TFLite output type: {out_det['dtype']}, shape: {out_det['shape']}")

correct = 0
total   = len(X_test_scaled)
y_pred_tflite = []
print(f"    Running {total} predictions...")
for i in range(total):
    inp = X_test_scaled[i:i+1]
    interpreter.set_tensor(inp_det['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(out_det['index'])[0]
    pred = int(np.argmax(out))
    y_pred_tflite.append(pred)
    if pred == y_test[i]:
        correct += 1
    if (i+1) % 500 == 0:
        print(f"    ... {i+1}/{total} ({100*(i+1)/total:.0f}%)", flush=True)

y_pred_tflite = np.array(y_pred_tflite)
tflite_acc = correct / total
print(f"\n    TFLite accuracy: {tflite_acc*100:.2f}%")
print(f"    Accuracy drop vs Keras: {(keras_acc - tflite_acc)*100:.2f}%")
print(confusion_matrix(y_test, y_pred_tflite))
print(classification_report(y_test, y_pred_tflite, target_names=CATEGORIES))

# Check if acceptable (< 2% drop)
if keras_acc - tflite_acc > 0.03:
    print("\n[WARNING] Accuracy drop > 3%! Consider re-training or increasing representative dataset.")
else:
    print(f"\n[OK] Accuracy drop is acceptable ({(keras_acc-tflite_acc)*100:.2f}%).")

# Generate model.h header
print(f"\n[8] Generating C++ deployment header: {OUT_HEADER}")
import librosa

hamming = np.hamming(N_FFT).astype(np.float32)
mel_fb  = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000).T
n_frames = (int(SAMPLE_RATE * DURATION) - N_FFT) // HOP_LENGTH + 1

with open(OUT_HEADER, 'w') as f:
    f.write("/*\n")
    f.write(" * model.h - Siren Classifier Deployment Header\n")
    f.write(" * Generated automatically by fix_quant_export.py\n")
    f.write(f" * TFLite size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.2f} KB)\n")
    f.write(f" * Keras accuracy (test): {keras_acc*100:.2f}%\n")
    f.write(f" * TFLite accuracy (test): {tflite_acc*100:.2f}%\n")
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

    # Hamming window
    f.write(f"const float HAMMING_WINDOW[{N_FFT}] PROGMEM = {{\n")
    for i, val in enumerate(hamming):
        if i % 8 == 0:
            f.write("  ")
        f.write(f"{val:.8f}f")
        if i < N_FFT - 1:
            f.write(", ")
        if (i + 1) % 8 == 0 or i == N_FFT - 1:
            f.write("\n")
    f.write("};\n\n")

    # Mel filterbank
    f.write(f"const float MEL_FILTERBANK[{N_MELS}][{N_FFT//2+1}] PROGMEM = {{\n")
    for m in range(N_MELS):
        f.write("  {")
        row = mel_fb[:, m]
        f.write(", ".join([f"{val:.8f}f" for val in row]))
        f.write("}")
        if m < N_MELS - 1:
            f.write(",\n")
        else:
            f.write("\n")
    f.write("};\n\n")

    # MEL_MEAN / MEL_STD
    f.write(f"const float MEL_MEAN[{N_MELS}] PROGMEM = {{\n  ")
    f.write(", ".join([f"{val:.8f}f" for val in global_mean]))
    f.write("\n};\n\n")

    f.write(f"const float MEL_STD[{N_MELS}] PROGMEM = {{\n  ")
    f.write(", ".join([f"{val:.8f}f" for val in global_std]))
    f.write("\n};\n\n")

    # Model data
    f.write("// Align model array for TFLite Micro\n")
    f.write("#ifdef __has_attribute\n")
    f.write("#define MODEL_ALIGN __attribute__((aligned(4)))\n")
    f.write("#else\n")
    f.write("#define MODEL_ALIGN\n")
    f.write("#endif\n\n")

    f.write(f"const unsigned int siren_model_data_len = {len(tflite_model)};\n\n")
    f.write("const unsigned char siren_model_data[] MODEL_ALIGN = {\n")
    for i, val in enumerate(tflite_model):
        if i % 12 == 0:
            f.write("  ")
        f.write(f"0x{val:02x}")
        if i < len(tflite_model) - 1:
            f.write(", ")
        if (i + 1) % 12 == 0 or i == len(tflite_model) - 1:
            f.write("\n")
    f.write("};\n\n")
    f.write("#endif // MODEL_H\n")

print(f"    Saved model.h ({os.path.getsize(OUT_HEADER)//1024} KB)")
print("\n[SUCCESS] Export complete!")
print(f"  TFLite  : {OUT_TFLITE}")
print(f"  model.h : {OUT_HEADER}")
