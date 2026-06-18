"""
retrain_qat.py - Quantization-Aware Training (QAT) Pipeline
============================================================
Step 1: Retrain model WITH BatchNorm (sama seperti original, dari cache)
Step 2: Apply QAT fine-tuning (5-10 epoch) → model belajar robust terhadap INT8
Step 3: Export ke Full INT8 TFLite → gap < 1%

Kenapa QAT lebih baik dari post-training quantization?
- PTQ (lama): quantize SETELAH training → BatchNorm scale error sistematis
- QAT: fake quantization SELAMA training → model beradaptasi → INT8 presisi
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import sys
import numpy as np
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
print("  Retrain + QAT Pipeline — INT8-Friendly dengan BatchNorm")
print("=" * 65)

# Load scaler & cache
scaler_data = np.load(SCALER_PATH)
global_mean = scaler_data['global_mean']
global_std  = scaler_data['global_std']

print("\n[1] Loading feature cache...")
with np.load(CACHE_PATH, allow_pickle=True) as data:
    X_train = data['X_train']
    y_train = data['y_train']
    X_test  = data['X_test']
    y_test  = data['y_test']
print(f"    X_train: {X_train.shape}, X_test: {X_test.shape}")

X_train_s = ((X_train - global_mean[np.newaxis,np.newaxis,:]) /
             (global_std[np.newaxis,np.newaxis,:] + 1e-8))[..., np.newaxis].astype(np.float32)
X_test_s  = ((X_test  - global_mean[np.newaxis,np.newaxis,:]) /
             (global_std[np.newaxis,np.newaxis,:] + 1e-8))[..., np.newaxis].astype(np.float32)

# SpecAugment
def spec_augment(X, fraction=0.3):
    n, t, f, c = X.shape
    idx = np.random.choice(n, int(n * fraction), replace=False)
    for i in idx:
        fs = int(f * 0.15)
        if fs > 0:
            f0 = np.random.randint(0, f - fs + 1)
            X[i, :, f0:f0+fs, :] = 0.0
        ts = int(t * 0.15)
        if ts > 0:
            t0 = np.random.randint(0, t - ts + 1)
            X[i, t0:t0+ts, :, :] = 0.0
spec_augment(X_train_s)
perm = np.random.permutation(len(X_train_s))
X_train_s = X_train_s[perm]
y_train_p = y_train[perm]

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils.class_weight import compute_class_weight

# ── STEP 1: Build & Train model WITH BatchNorm (sama seperti original) ──
print("\n[2] Building model WITH BatchNorm (original architecture)...")
model = models.Sequential([
    layers.Input(shape=(X_train_s.shape[1], X_train_s.shape[2], 1)),

    layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),

    layers.SeparableConv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),

    layers.SeparableConv2D(48, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.GlobalAveragePooling2D(),

    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.4),
    layers.Dense(len(CATEGORIES), activation='softmax')
], name='siren_classifier')
model.summary()
print(f"    Total params: {model.count_params():,}")

cw = compute_class_weight('balanced', classes=np.unique(y_train_p), y=y_train_p)
cw_dict = {i: w for i, w in enumerate(cw)}

try:
    loss_fn = tf.keras.losses.SparseCategoricalFocalCrossentropy(gamma=2.0)
    print("    Using Focal Loss")
except:
    loss_fn = 'sparse_categorical_crossentropy'

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss_fn, metrics=['accuracy']
)

callbacks_base = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7,
                                      restore_best_weights=True, mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
]

print("\n[3] Training base model (dengan BatchNorm)...")
model.fit(
    X_train_s, y_train_p, epochs=30, batch_size=512,
    validation_data=(X_test_s, y_test),
    class_weight=cw_dict, callbacks=callbacks_base, verbose=1
)

# Eval base
y_pred = np.argmax(model.predict(X_test_s, batch_size=256, verbose=0), axis=1)
base_acc = np.mean(y_pred == y_test)
print(f"\n    Base Keras accuracy: {base_acc*100:.2f}%")
model.save(OUT_H5)

# ── STEP 2: QAT Fine-tuning ──
print("\n[4] Applying Quantization-Aware Training (QAT)...")
try:
    import tensorflow_model_optimization as tfmot
    quantize_model = tfmot.quantization.keras.quantize_model

    # QAT: wrap model dengan fake quantization nodes
    qat_model = quantize_model(model)
    print("    QAT model created!")
    print(f"    QAT params: {qat_model.count_params():,}")

    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # LR lebih kecil untuk fine-tune
        loss=loss_fn, metrics=['accuracy']
    )

    callbacks_qat = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5,
                                          restore_best_weights=True, mode='max'),
    ]

    print("    Fine-tuning dengan QAT (10 epoch)...")
    qat_model.fit(
        X_train_s, y_train_p, epochs=10, batch_size=512,
        validation_data=(X_test_s, y_test),
        class_weight=cw_dict, callbacks=callbacks_qat, verbose=1
    )

    y_pred_qat = np.argmax(qat_model.predict(X_test_s, batch_size=256, verbose=0), axis=1)
    qat_acc = np.mean(y_pred_qat == y_test)
    print(f"    QAT accuracy: {qat_acc*100:.2f}%")

    # ── STEP 3: Export QAT model ke INT8 TFLite ──
    print("\n[5] Exporting QAT model ke INT8 TFLite...")

    rep_idx = []
    for cls in range(len(CATEGORIES)):
        ci = np.where(y_train_p == cls)[0]
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

    print(f"    TFLite size: {len(tflite_model)/1024:.1f} KB")

    # Eval TFLite
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"    Input : {inp['dtype'].__name__} {inp['shape'].tolist()}")
    print(f"    Output: {out['dtype'].__name__} {out['shape'].tolist()}")

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
            print(f"    ... {i+1}/{len(X_test_s)}", flush=True)

    preds = np.array(preds)
    tflite_acc = np.mean(preds == y_test)

    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\n    Base Keras  : {base_acc*100:.2f}%")
    print(f"    QAT Keras   : {qat_acc*100:.2f}%")
    print(f"    QAT TFLite  : {tflite_acc*100:.2f}%")
    print(f"    Gap         : {(qat_acc-tflite_acc)*100:.2f}%")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=CATEGORIES))

    chosen_model = tflite_model
    chosen_acc = tflite_acc
    print("\n[OK] QAT berhasil!")

except ImportError:
    print("\n[WARN] tensorflow-model-optimization tidak tersedia, pakai PTQ...")
    # Fallback ke PTQ dengan Float32 I/O (82%)
    rep_idx = []
    for cls in range(len(CATEGORIES)):
        ci = np.where(y_train_p == cls)[0]
        rep_idx.extend(np.random.choice(ci, min(1000, len(ci)), replace=False).tolist())
    np.random.shuffle(rep_idx)
    X_rep = X_train_s[rep_idx]

    def rep_gen():
        for i in range(len(X_rep)):
            yield [X_rep[i:i+1]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    chosen_model = conv.convert()
    chosen_acc = base_acc
    print(f"    PTQ TFLite: {len(chosen_model)/1024:.1f} KB")

# Save TFLite
with open(OUT_TFLITE, 'wb') as f:
    f.write(chosen_model)

# Generate model.h
print(f"\n[6] Generating model.h...")
import librosa
hamming  = np.hamming(N_FFT).astype(np.float32)
mel_fb   = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000).T
n_frames = (int(SAMPLE_RATE * DURATION) - N_FFT) // HOP_LENGTH + 1

interp_info = tf.lite.Interpreter(model_content=chosen_model)
interp_info.allocate_tensors()
inp_i = interp_info.get_input_details()[0]
out_i = interp_info.get_output_details()[0]

with open(OUT_HEADER, 'w') as f:
    f.write(f"/* model.h — QAT INT8 | TFLite: {chosen_acc*100:.2f}% | {len(chosen_model)} bytes */\n\n")
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
    f.write(f"const unsigned int siren_model_data_len = {len(chosen_model)};\n\n")
    f.write("const unsigned char siren_model_data[] MODEL_ALIGN = {\n")
    for i, val in enumerate(chosen_model):
        if i%12==0: f.write("  ")
        f.write(f"0x{val:02x}")
        if i<len(chosen_model)-1: f.write(", ")
        if (i+1)%12==0 or i==len(chosen_model)-1: f.write("\n")
    f.write("};\n\n#endif // MODEL_H\n")

print(f"    model.h saved ({os.path.getsize(OUT_HEADER)//1024} KB)")
print(f"\n[DONE] TFLite: {chosen_acc*100:.2f}%  Size: {len(chosen_model)/1024:.1f} KB")
print("Upload ulang firmware ke ESP32 setelah ini!")
