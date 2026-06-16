import os
import random
import numpy as np
import joblib
import tensorflow as tf

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset_dir = r"c:\Users\ASUS\Videos\DATASET"
scaler_path = os.path.join(dataset_dir, "siren_scaler.joblib")
model_path = os.path.join(dataset_dir, "siren_classifier_model.h5")

try:
    from predict_siren import extract_chunks_features
except ImportError:
    print("Error: Could not import predict_siren.py")
    exit(1)

print("Memuat AI Model untuk Audit (Mencari file yang suaranya salah)...")
model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

# Label saat training (berdasarkan train_siren_model.py)
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

def predict_audio(filepath):
    try:
        features = extract_chunks_features(filepath)
        features_scaled = scaler.transform(features)
        features_input = features_scaled.reshape(len(features_scaled), -1, 1)
        
        all_probs = model.predict(features_input, verbose=0)
        probs = np.mean(all_probs, axis=0)
        
        best_idx = np.argmax(probs)
        confidence = probs[best_idx] * 100
        return best_idx, confidence
    except Exception as e:
        return -1, 0

def audit_folder(folder_name, true_idx, sample_size=100):
    folder_path = os.path.join(dataset_dir, folder_name)
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    random.shuffle(files)
    
    suspects = []
    print(f"\nMengaudit {sample_size} file acak dari {folder_name} menggunakan AI...")
    
    for i, f in enumerate(files[:sample_size]):
        filepath = os.path.join(folder_path, f)
        pred_idx, conf = predict_audio(filepath)
        
        # Jika AI yakin >80% bahwa ini bukan kelas aslinya, maka dicatat sebagai MENCURIGAKAN
        if pred_idx != -1 and pred_idx != true_idx and conf > 80.0:
            predicted_label = CATEGORIES[pred_idx] if pred_idx < len(CATEGORIES) else "UNKNOWN"
            suspects.append((f, predicted_label, conf))
            
    return suspects

# Kita akan audit 200 file acak dari folder AMBULANCE karena keluhan "Ambulance dikira Polisi"
amb_suspects = audit_folder("AMBULANCE", 0, sample_size=200)
# Audit juga 200 file acak dari POLICE
pol_suspects = audit_folder("POLICE", 3, sample_size=200)

print("\n" + "="*60)
print("HASIL AUDIT ISI SUARA DENGAN AI:")
print("="*60)
if len(amb_suspects) == 0 and len(pol_suspects) == 0:
    print("LUAR BIASA! AI tidak menemukan audio yang salah isi (Salah Kamar) pada sampel ini.")
    print("Dataset ini sepertinya sangat murni.")
else:
    if len(amb_suspects) > 0:
        print(f"Ditemukan {len(amb_suspects)} file di folder AMBULANCE yang sebenarnya bersuara lain:")
        for f, p, c in amb_suspects:
            print(f"  [X] {f} --> Terdengar seperti {p} (Yakin {c:.1f}%)")
            
    if len(pol_suspects) > 0:
        print(f"\nDitemukan {len(pol_suspects)} file di folder POLICE yang sebenarnya bersuara lain:")
        for f, p, c in pol_suspects:
            print(f"  [X] {f} --> Terdengar seperti {p} (Yakin {c:.1f}%)")

print("\nAudit Selesai.")
