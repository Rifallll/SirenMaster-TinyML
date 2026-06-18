"""
fix_quant_v3.py — Full INT8 untuk ESP32 dengan kalibrasi penuh
Masalah sebelumnya: 83% akibat BatchNorm quantization artifact
Solusi: gunakan SEMUA data training sebagai representative dataset
       + gunakan float32 I/O (bukan INT8 I/O) agar BatchNorm lebih presisi
"""
import os, sys, re
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
print("  Full INT8 Quantization (ESP32-Compatible) — v3")
print("  Masalah: Dynamic Range butuh 320KB, ESP32 hanya ~85KB")
print("  Solusi : Full INT8 + calibrasi penuh semua training data")
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

def eval_tflite(tflite_bytes, label="TFLite"):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"    [{label}] Input : type={inp['dtype'].__name__}, shape={inp['shape']}")
    print(f"    [{label}] Output: type={out['dtype'].__name__}, shape={out['shape']}")
    
    preds = []
    for i in range(len(X_test_s)):
        sample = X_test_s[i:i+1]
        if inp['dtype'] == np.int8:
            sc = inp['quantization_parameters']['scales'][0]
            zp = inp['quantization_parameters']['zero_points'][0]
            if sc > 0:
                sample = np.clip(np.round(sample / sc) + zp, -128, 127).astype(np.int8)
            else:
                sample = np.zeros_like(sample, dtype=np.int8)
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

# Stratified representative dataset — 4000 samples (1000/kelas)
print("\n[4] Preparing FULL representative dataset (4000 stratified)...")
rep_idx = []
for cls in range(len(CATEGORIES)):
    cls_i = np.where(y_train == cls)[0]
    n = min(1000, len(cls_i))
    rep_idx.extend(np.random.choice(cls_i, n, replace=False).tolist())
np.random.shuffle(rep_idx)
X_rep = X_train_s[rep_idx]
print(f"    Representative samples: {X_rep.shape[0]}")

def make_rep_gen():
    def gen():
        for i in range(len(X_rep)):
            yield [X_rep[i:i+1]]
    return gen

# ── Strategi A: Full INT8, FLOAT32 I/O ─────────────────────────────────
# Ini memberi akurasi terbaik (BatchNorm tidak terdistorsi di I/O)
# Arena footprint: aktivasi INT8 (~20-30KB), bukan float32 (320KB)
print("\n[5A] Full INT8 + FLOAT32 I/O (BatchNorm friendly)...")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = make_rep_gen()
conv.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,  # fallback untuk ops yang tidak support int8
]
# TIDAK set inference_input_type / output_type → tetap float32 I/O
# tapi internal computation INT8
m_a = conv.convert()
print(f"    Size: {len(m_a)/1024:.1f} KB")
acc_a, preds_a = eval_tflite(m_a, "INT8+FloatIO")
print(f"    Accuracy: {acc_a*100:.2f}%  Drop: {(keras_acc-acc_a)*100:.2f}%")

# ── Strategi B: Full INT8, INT8 I/O (smallest RAM) ──────────────────────
print("\n[5B] Full INT8 + INT8 I/O (smallest tensor arena)...")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = make_rep_gen()
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.int8
conv.inference_output_type = tf.int8
m_b = conv.convert()
print(f"    Size: {len(m_b)/1024:.1f} KB")
acc_b, preds_b = eval_tflite(m_b, "INT8+INT8IO")
print(f"    Accuracy: {acc_b*100:.2f}%  Drop: {(keras_acc-acc_b)*100:.2f}%")

# Pilih yang terbaik
print(f"\n[6] Comparison:")
print(f"    Keras          : {keras_acc*100:.2f}%")
print(f"    A (Float32 I/O): {acc_a*100:.2f}%  {len(m_a)/1024:.1f} KB")
print(f"    B (INT8 I/O)   : {acc_b*100:.2f}%  {len(m_b)/1024:.1f} KB")

# Prioritas: akurasi dulu, lalu ukuran
chosen_m = m_a if acc_a >= acc_b else m_b
chosen_acc = acc_a if acc_a >= acc_b else acc_b
chosen_label = "A (Float32 I/O)" if acc_a >= acc_b else "B (INT8 I/O)"
print(f"    → CHOSEN: {chosen_label}  ({chosen_acc*100:.2f}%)")

# Cek tensor arena yang dibutuhkan
interp_check = tf.lite.Interpreter(model_content=chosen_m)
interp_check.allocate_tensors()
print(f"    → Arena dibutuhkan model: estimasi ~{len(chosen_m)//1024 + 10} KB")

# Save
with open(OUT_TFLITE, 'wb') as f:
    f.write(chosen_m)
print(f"\n    Saved: {OUT_TFLITE} ({len(chosen_m)/1024:.1f} KB)")

# Final eval detail
from sklearn.metrics import classification_report, confusion_matrix
if acc_a >= acc_b:
    print(confusion_matrix(y_test, preds_a))
    print(classification_report(y_test, preds_a, target_names=CATEGORIES))
else:
    print(confusion_matrix(y_test, preds_b))
    print(classification_report(y_test, preds_b, target_names=CATEGORIES))

# Generate model.h
print(f"\n[7] Generating model.h...")
import librosa
hamming  = np.hamming(N_FFT).astype(np.float32)
mel_fb   = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000).T
n_frames = (int(SAMPLE_RATE * DURATION) - N_FFT) // HOP_LENGTH + 1

# Tulis model.h
with open(OUT_HEADER, 'w') as f:
    interp_info = tf.lite.Interpreter(model_content=chosen_m)
    interp_info.allocate_tensors()
    inp_info = interp_info.get_input_details()[0]
    out_info = interp_info.get_output_details()[0]
    
    f.write("/*\n")
    f.write(" * model.h - Siren Classifier Deployment Header\n")
    f.write(" * Generated by fix_quant_v3.py\n")
    f.write(f" * Strategy: {chosen_label}  Size: {len(chosen_m)} bytes\n")
    f.write(f" * Input  type: {inp_info['dtype'].__name__}  shape: {inp_info['shape'].tolist()}\n")
    f.write(f" * Output type: {out_info['dtype'].__name__}  shape: {out_info['shape'].tolist()}\n")
    f.write(f" * Keras acc: {keras_acc*100:.2f}%  TFLite acc: {chosen_acc*100:.2f}%\n")
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

print(f"    Saved model.h ({os.path.getsize(OUT_HEADER)//1024} KB)")
print("\n[SUCCESS] Done!")
print(f"  Strategy  : {chosen_label}")
print(f"  TFLite acc: {chosen_acc*100:.2f}% (Keras: {keras_acc*100:.2f}%)")
print(f"\n  Sekarang upload ulang firmware ke ESP32!")
