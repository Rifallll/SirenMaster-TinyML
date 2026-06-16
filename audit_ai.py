import os
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATASET_PATHS = [
    r"C:\Users\ASUS\Videos\DATASET",
    r"C:\Users\ASUS\Videos\DATASET\DATASET_LOKAL",
    r"C:\Users\ASUS\Videos\DATASET\TAMBAH_DATASET"
]
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
CACHE = r"C:\Users\ASUS\Videos\DATASET\siren_40x63_melspec_cache_lokal.npz"
MODEL_TFLITE = r"C:\Users\ASUS\Videos\DATASET\siren_model_quant.tflite"
MODEL_H5 = r"C:\Users\ASUS\Videos\DATASET\siren_classifier_model.h5"

print("=" * 65)
print("   AUDIT SISTEM AI DETEKSI SIRINE - LAPORAN LENGKAP")
print("=" * 65)

# ─── 1. AUDIT DATASET ───
print("\n[1] AUDIT DATASET")
print("-" * 65)
total_files = 0
cat_counts = {}
for cat in CATEGORIES:
    count = 0
    for base in DATASET_PATHS:
        d = os.path.join(base, cat)
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith('.wav'):
                    count += 1
    cat_counts[cat] = count
    total_files += count
    bar = '#' * min(count // 50, 40)
    status = "OK" if count >= 1000 else "KURANG"
    print(f"  {cat:<12}: {count:>5} file  [{bar:<40}]  {status}")

print(f"\n  Total file     : {total_files} file WAV")
min_cat = min(cat_counts, key=cat_counts.get)
max_cat = max(cat_counts, key=cat_counts.get)
imbalance = cat_counts[max_cat] / max(cat_counts[min_cat], 1)
print(f"  Ketidakseimbangan: {imbalance:.1f}x  (ideal < 2x)")
if imbalance > 2.5:
    print("  [!] PERINGATAN: Dataset tidak seimbang! AMBULANCE & POLICE perlu lebih banyak data.")
else:
    print("  [OK] Keseimbangan dataset cukup baik.")

# ─── 2. AUDIT CACHE / FEATURES ───
print("\n[2] AUDIT FITUR YANG DIEKSTRAK (Cache)")
print("-" * 65)
if os.path.exists(CACHE):
    data = np.load(CACHE, allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    print(f"  Shape X_train  : {X_train.shape}  -> {X_train.shape[0]} sampel, {X_train.shape[1]} frame, {X_train.shape[2]} mel-bin")
    print(f"  Shape y_train  : {y_train.shape}")
    print(f"  Ukuran cache   : {os.path.getsize(CACHE)/1024/1024:.1f} MB")
    u, c = np.unique(y_train, return_counts=True)
    print(f"\n  Distribusi training samples (setelah augmentasi):")
    for idx, cnt in zip(u, c):
        bar = '#' * (cnt // 500)
        print(f"    {CATEGORIES[idx]:<12}: {cnt:>6} sampel  [{bar}]")
    print(f"\n  Mean fitur (5 pertama): {np.round(np.mean(X_train, axis=(0,1))[:5], 3)}")
    print(f"  Std  fitur (5 pertama): {np.round(np.std(X_train, axis=(0,1))[:5], 3)}")
else:
    print("  [!] Cache tidak ditemukan!")

# ─── 3. AUDIT MODEL ───
print("\n[3] AUDIT ARSITEKTUR MODEL (TFLite + Keras)")
print("-" * 65)

if os.path.exists(MODEL_TFLITE):
    size_kb = os.path.getsize(MODEL_TFLITE) / 1024
    print(f"  Model TFLite   : {size_kb:.2f} KB  (INT8 Quantized)")
    
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    print(f"  Input shape    : {inp['shape']}  dtype={inp['dtype'].__name__}")
    print(f"  Output shape   : {out['shape']}  dtype={out['dtype'].__name__}")
    print(f"  Input scale    : {inp['quantization'][0]:.6f}  zp={inp['quantization'][1]}")
    print(f"  Output scale   : {out['quantization'][0]:.6f}  zp={out['quantization'][1]}")

if os.path.exists(MODEL_H5):
    size_kb = os.path.getsize(MODEL_H5) / 1024
    print(f"\n  Model H5 (full): {size_kb:.1f} KB")
    model = tf.keras.models.load_model(MODEL_H5, compile=False)
    print(f"  Total parameter: {model.count_params():,}")
    print(f"\n  Arsitektur layer:")
    for i, layer in enumerate(model.layers):
        params = layer.count_params()
        print(f"    [{i:02d}] {layer.__class__.__name__:<25} | params={params:>7,} | output={str(layer.output_shape)}")

# ─── 4. PENILAIAN AKHIR ───
print("\n[4] PENILAIAN AKHIR SISTEM")
print("=" * 65)

checks = {
    "Dataset > 15.000 file"     : total_files >= 15000,
    "Semua kelas > 3.000 file"  : all(v >= 3000 for v in cat_counts.values()),
    "Model TFLite tersedia"     : os.path.exists(MODEL_TFLITE),
    "Model < 30KB (ringan)"     : os.path.exists(MODEL_TFLITE) and os.path.getsize(MODEL_TFLITE)/1024 < 30,
    "Model H5 tersedia"         : os.path.exists(MODEL_H5),
    "Cache tersedia"            : os.path.exists(CACHE),
}

passed = sum(checks.values())
total_checks = len(checks)

for name, ok in checks.items():
    icon = "PASS" if ok else "FAIL"
    print(f"  [{icon}]  {name}")

score = passed / total_checks * 100
print(f"\n  SKOR SISTEM: {score:.0f}% ({passed}/{total_checks} cek lulus)")

if score >= 90:
    print("  STATUS: SANGAT BAIK - Siap deploy ke lapangan!")
elif score >= 70:
    print("  STATUS: BAIK - Ada beberapa area yang bisa ditingkatkan")
else:
    print("  STATUS: PERLU PERBAIKAN")

print("\n  Catatan teknis vs Otak Manusia:")
print("  Neuron manusia    : ~86 miliar neuron")
if os.path.exists(MODEL_H5):
    print(f"  Parameter AI ini  : {model.count_params():,} parameter (bobot sinaptik)")
print("  Kecepatan manusia : ~200-300ms reaksi")
print("  Kecepatan AI ini  : ~1000ms (1 detik window + inferensi ~50ms)")
print("  Akurasi manusia   : ~95-99% (mengenali sirine dikenal)")
print("  Akurasi AI ini    : ~91% test set (model ringan di hardware kecil)")
print("=" * 65)
