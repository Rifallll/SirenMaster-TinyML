import os
import librosa
import numpy as np

dataset_dir = r"C:\Users\ASUS\Videos\DATASET"
classes = ["AMBULANCE", "FIRETRUCK", "NORMAL", "POLICE"]

print("=== AUDIT DATASET SIRENMASTER ===")
for c in classes:
    folder = os.path.join(dataset_dir, c)
    if not os.path.exists(folder):
        print(f"[!] Folder tidak ditemukan: {c}")
        continue
    
    files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    print(f"\n-> Kategori: {c} | Jumlah File: {len(files)}")
    
    if len(files) == 0:
        continue
    
    # Ambil sample maksimal 50 file untuk mempercepat audit (diambil secara acak)
    sample_files = list(np.random.choice(files, min(50, len(files)), replace=False))
    durations = []
    sample_rates = set()
    
    for f in sample_files:
        try:
            path = os.path.join(folder, f)
            y, sr = librosa.load(path, sr=None)
            dur = len(y) / sr
            durations.append(dur)
            sample_rates.add(sr)
        except Exception as e:
            pass
            
    if durations:
        print(f"  Rata-rata Durasi: {np.mean(durations):.2f} detik")
        print(f"  Durasi Min-Max  : {np.min(durations):.2f} - {np.max(durations):.2f} detik")
        print(f"  Sample Rate     : {', '.join(map(str, sample_rates))} Hz")
        
print("\nAudit Selesai. Data siap untuk diuji lebih lanjut.")
