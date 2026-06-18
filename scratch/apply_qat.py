import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # WAJIB untuk kompatibilitas QAT

import sys
import numpy as np
import tensorflow as tf
# Pastikan pakai tf_keras agar didukung oleh tensorflow_model_optimization
import tf_keras as keras

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATASET_BASE = r"C:\Users\ASUS\Videos\DATASET"
CACHE_PATH   = os.path.join(DATASET_BASE, "siren_40x249_melspec_cache_lokal.npz")
SCALER_PATH  = os.path.join(DATASET_BASE, "siren_scaler.npz")
OUT_H5       = os.path.join(DATASET_BASE, "siren_classifier_model.h5")
OUT_TFLITE   = os.path.join(DATASET_BASE, "siren_model_quant.tflite")
OUT_HEADER   = os.path.join(DATASET_BASE, "sirenmaster_main", "model.h")
CATEGORIES   = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
SAMPLE_RATE  = 8000; N_FFT = 256; HOP_LENGTH = 128
N_MELS = 40; N_MFCC = 13; DURATION = 4.0

print("=" * 65)
print("  Melanjutkan QAT Fine-tuning dari Model 89%...")
print("=" * 65)

scaler_data = np.load(SCALER_PATH)
global_mean = scaler_data['global_mean']
global_std  = scaler_data['global_std']

print("[1] Loading data...")
with np.load(CACHE_PATH, allow_pickle=True) as data:
    X_train = data['X_train']
    y_train = data['y_train']
    X_test  = data['X_test']
    y_test  = data['y_test']

X_train_s = ((X_train - global_mean[np.newaxis,np.newaxis,:]) /
             (global_std[np.newaxis,np.newaxis,:] + 1e-8))[..., np.newaxis].astype(np.float32)
X_test_s  = ((X_test  - global_mean[np.newaxis,np.newaxis,:]) /
             (global_std[np.newaxis,np.newaxis,:] + 1e-8))[..., np.newaxis].astype(np.float32)

from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = {i: w for i, w in enumerate(cw)}

print("[2] Loading Base Model...")
model = keras.models.load_model(OUT_H5, compile=False)

import tensorflow_model_optimization as tfmot
quantize_model = tfmot.quantization.keras.quantize_model

print("[3] Applying QAT...")
qat_model = quantize_model(model)
qat_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("[4] Fine-tuning dengan QAT (5 Epoch)...")
qat_model.fit(
    X_train_s, y_train, epochs=5, batch_size=512,
    validation_data=(X_test_s, y_test),
    class_weight=cw_dict, verbose=1
)

y_pred_qat = np.argmax(qat_model.predict(X_test_s, batch_size=256, verbose=0), axis=1)
qat_acc = np.mean(y_pred_qat == y_test)
print(f"    QAT accuracy: {qat_acc*100:.2f}%")

print("\n[5] Exporting QAT model ke INT8 TFLite...")
rep_idx = []
for cls in range(len(CATEGORIES)):
    ci = np.where(y_train == cls)[0]
    rep_idx.extend(np.random.choice(ci, min(1000, len(ci)), replace=False).tolist())
np.random.shuffle(rep_idx)
X_rep = X_train_s[rep_idx]

def rep_gen():
    for i in range(len(X_rep)):
        yield [X_rep[i:i+1]]

conv = tf.lite.TFLiteConverter.from_keras_model(qat_model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_gen
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = conv.convert()

interp = tf.lite.Interpreter(model_content=tflite_model)
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

preds = np.array(preds)
tflite_acc = np.mean(preds == y_test)

from sklearn.metrics import classification_report
print(f"    QAT TFLite  : {tflite_acc*100:.2f}%")
print(classification_report(y_test, preds, target_names=CATEGORIES))

with open(OUT_TFLITE, 'wb') as f:
    f.write(tflite_model)

import librosa
hamming  = np.hamming(N_FFT).astype(np.float32)
mel_fb   = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000).T
n_frames = (int(SAMPLE_RATE * DURATION) - N_FFT) // HOP_LENGTH + 1

with open(OUT_HEADER, 'w') as f:
    f.write(f"/* model.h — QAT INT8 | TFLite: {tflite_acc*100:.2f}% | {len(tflite_model)} bytes */\n\n")
    f.write("#ifndef MODEL_H\n#define MODEL_H\n\n#include <pgmspace.h>\n\n")
    f.write(f"#define SAMPLE_RATE_HZ   {SAMPLE_RATE}\n#define N_FFT_SIZE       {N_FFT}\n")
    f.write(f"#define N_HOP_LENGTH     {HOP_LENGTH}\n#define N_MFCC_COEFF     {N_MFCC}\n")
    f.write(f"#define N_MEL_FILTERS    {N_MELS}\n#define N_TIME_FRAMES    {n_frames}\n")
    f.write(f"#define N_FFT_BINS       {N_FFT//2+1}\n#define AUDIO_SAMPLES    {int(SAMPLE_RATE*DURATION)}\n")
    f.write(f"#define NUM_CLASSES      {len(CATEGORIES)}\n\n")
    f.write(f"const float HAMMING_WINDOW[{N_FFT}] PROGMEM = {{\n")
    for i, v in enumerate(hamming):
        if i%8==0: f.write("  ")
        f.write(f"{v:.8f}f")
        if i<N_FFT-1: f.write(", ")
        if (i+1)%8==0 or i==N_FFT-1: f.write("\n")
    f.write("};\n\n")
    f.write(f"const float MEL_FILTERBANK[{N_MELS}][{N_FFT//2+1}] PROGMEM = {{\n")
    for mi in range(N_MELS):
        f.write("  {" + ",".join([f"{v:.8f}f" for v in mel_fb[:,mi]]) + "}")
        f.write(",\n" if mi<N_MELS-1 else "\n")
    f.write("};\n\n")
    f.write(f"const float MEL_MEAN[{N_MELS}] PROGMEM = {{\n  ")
    f.write(",".join([f"{v:.8f}f" for v in global_mean])); f.write("\n};\n\n")
    f.write(f"const float MEL_STD[{N_MELS}] PROGMEM = {{\n  ")
    f.write(",".join([f"{v:.8f}f" for v in global_std])); f.write("\n};\n\n")
    f.write("#ifdef __has_attribute\n#define MODEL_ALIGN __attribute__((aligned(4)))\n#else\n#define MODEL_ALIGN\n#endif\n\n")
    f.write(f"const unsigned int siren_model_data_len = {len(tflite_model)};\n\n")
    f.write("const unsigned char siren_model_data[] MODEL_ALIGN = {\n")
    for i, val in enumerate(tflite_model):
        if i%12==0: f.write("  ")
        f.write(f"0x{val:02x}")
        if i<len(tflite_model)-1: f.write(", ")
        if (i+1)%12==0 or i==len(tflite_model)-1: f.write("\n")
    f.write("};\n\n#endif // MODEL_H\n")

print("[SUCCESS] model.h telah di-generate dengan QAT INT8 Model.")
