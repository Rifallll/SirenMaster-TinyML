"""
audit_akurasi_model_ai.py
=========================
Audit komprehensif akurasi model TFLite terkuantisasi (INT8) yang digunakan
di ESP32 (model.tflite) terhadap sampel bersih dataset (Ambulance, Firetruck, Police, Normal).
Menghasilkan Matriks Kebingungan (Confusion Matrix) dan mengecek apakah ada salah kelas.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, glob, random, librosa, numpy as np
import tensorflow as tf

ROOT = r"C:\Users\ASUS\Videos\DATASET"
# Model yang sama persis dengan yang tertanam di ESP32 dan scaler-nya
TFLITE_PATH = os.path.join(ROOT, "04_Training_AI", "siren_model_quant.tflite")
if not os.path.exists(TFLITE_PATH):
    TFLITE_PATH = os.path.join(ROOT, "siren_model_quant.tflite")

SCALER_PATH = os.path.join(ROOT, "04_Training_AI", "siren_scaler.npz")

CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
NUM_CLASSES = len(CATEGORIES)

print("=" * 80)
print(" 🤖 AUDIT AKURASI & MATRIKS KEBINGUNGAN MODEL AI ESP32 (INT8 TFLITE)")
print("=" * 80)
print(f"[*] Menggunakan Model : {TFLITE_PATH}")

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_scale, in_zero = input_details[0]['quantization']
out_scale, out_zero = output_details[0]['quantization']
is_quant = (in_scale != 0.0)

# Load Scaler
scaler = np.load(SCALER_PATH)
global_mean = scaler['global_mean']
global_std = scaler['global_std']

def extract_feat(filepath):
    try:
        y, sr = librosa.load(filepath, sr=8000, duration=4.0)
    except Exception:
        return None
    if len(y) < 32000:
        y = np.pad(y, (0, 32000 - len(y)))
    else:
        y = y[:32000]

    n_frames = (32000 - 256) // 128 + 1
    hamming = np.hamming(256)
    mel_fb = librosa.filters.mel(sr=8000, n_fft=256, n_mels=40, fmin=0, fmax=4000)

    log_mel_frames = []
    for frame in range(n_frames):
        start = frame * 128
        fd = y[start:start+256].copy()
        fft_out = np.fft.rfft(fd * hamming, n=256)
        power = np.abs(fft_out) ** 2
        mel_e = np.dot(mel_fb, power)
        log_mel_frames.append(np.log(mel_e + 1e-9))

    feat = np.array(log_mel_frames, dtype=np.float32)
    feat = (feat - global_mean) / global_std
    return feat

# Matriks kebingungan: [Actual][Predicted]
conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
errors_found = []

# Audit 150 sampel acak representatif dari setiap kelas (Total 600 sampel)
SAMPLES_PER_CLASS = 150
random.seed(42)

for true_idx, category in enumerate(CATEGORIES):
    folder_path = os.path.join(ROOT, category)
    all_wavs = sorted(glob.glob(os.path.join(folder_path, "*.wav")))
    if not all_wavs:
        continue
        
    # Pilih 150 sampel secara acak namun konsisten (berdasarkan seed 42)
    sample_wavs = random.sample(all_wavs, min(SAMPLES_PER_CLASS, len(all_wavs)))
    print(f"\n[*] Menguji {len(sample_wavs)} sampel dari kelas [{category}]...")
    
    for filepath in sample_wavs:
        filename = os.path.basename(filepath)
        feat = extract_feat(filepath)
        if feat is None:
            continue
            
        # Reshape sesuai persis shape yang diminta oleh model TFLite
        expected_shape = input_details[0]['shape']
        feat = np.reshape(feat, expected_shape)
        
        if is_quant:
            input_data = np.round(feat / in_scale + in_zero).astype(np.int8)
        else:
            input_data = feat
            
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        if is_quant:
            probs = (output_data.astype(np.float32) - out_zero) * out_scale
        else:
            probs = output_data
            
        pred_idx = int(np.argmax(probs))
        conf_matrix[true_idx][pred_idx] += 1
        
        # Jika salah kelas (dan keyakinannya lumayan tinggi > 60%), catat
        if pred_idx != true_idx:
            errors_found.append((category, CATEGORIES[pred_idx], filename, probs[pred_idx]*100.0))

# --- LAPORAN MATRIKS KEBINGUNGAN ---
print("\n" + "=" * 80)
print(" 📊 HASIL MATRIKS KEBINGUNGAN (CONFUSION MATRIX) MODEL AI ESP32")
print("=" * 80)
print(f"      {'PRED ->':<10} | {'AMBULANCE':<10} | {'FIRETRUCK':<10} | {'NORMAL':<10} | {'POLICE':<10} | {'AKURASI':<8}")
print("-" * 80)

total_tested = np.sum(conf_matrix)
total_correct = np.trace(conf_matrix)

for i in range(NUM_CLASSES):
    row_sum = np.sum(conf_matrix[i])
    acc = (conf_matrix[i][i] / row_sum * 100.0) if row_sum > 0 else 0.0
    print(f" {CATEGORIES[i]:<12} | {conf_matrix[i][0]:<10d} | {conf_matrix[i][1]:<10d} | {conf_matrix[i][2]:<10d} | {conf_matrix[i][3]:<10d} | {acc:.1f}%")

overall_acc = (total_correct / total_tested * 100.0) if total_tested > 0 else 0.0
print("-" * 80)
print(f" 🎯 AKURASI KESELURUHAN (OVERALL ACCURACY) : {overall_acc:.2f}% ({total_correct}/{total_tested} Sampel Benar)")
print("=" * 80)

if errors_found:
    print(f"\n[*] Catatan Kasus Salah Kelas ({len(errors_found)} dari {total_tested} sampel uji):")
    for idx, (tru_c, prd_c, fn, conf) in enumerate(errors_found[:12], 1):
        print(f"  {idx:2d}. [Asli: {tru_c:<10}] {fn:<38} -> Ditebak: {prd_c:<10} (conf: {conf:.1f}%)")
    if len(errors_found) > 12:
        print(f"      ... dan {len(errors_found)-12} kasus minor lainnya.")
else:
    print("\n[🏆 SEMPURNA] Tidak ada satu pun kasus salah kelas dari seluruh sampel uji!")
print("=" * 80)
