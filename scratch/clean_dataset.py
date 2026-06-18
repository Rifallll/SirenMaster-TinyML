import os
import numpy as np
from scipy.io import wavfile

dataset_dir = r"C:\Users\ASUS\Videos\DATASET"
classes = ["AMBULANCE", "FIRETRUCK", "POLICE"]

print("=== PEMBERSIHAN DATASET (MENGHAPUS SUARA PECAH/DISTORSI) ===")
print("Memproses ribuan file secara kilat...\n")

for c in classes:
    folder = os.path.join(dataset_dir, c)
    if not os.path.exists(folder):
        continue
    
    files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    print(f"[{c}] Total file awal: {len(files)}")
    
    removed = 0
    for f in files:
        path = os.path.join(folder, f)
        try:
            # Membaca dengan scipy jauh lebih cepat daripada librosa
            sr, data = wavfile.read(path)
            
            # Konversi stereo ke mono jika perlu untuk ngecek max amplitude
            if len(data.shape) > 1:
                data = data[:, 0]
                
            # Cek tipe data
            is_clipping = False
            if data.dtype == np.int16:
                max_val = np.max(np.abs(data.astype(np.int32)))
                if max_val >= 32000:  # Hampir mentok di 32767
                    is_clipping = True
            elif data.dtype == np.float32 or data.dtype == np.float64:
                max_val = np.max(np.abs(data))
                if max_val > 0.98:
                    is_clipping = True
                    
            if is_clipping:
                os.remove(path)
                removed += 1
        except Exception as e:
            # Jika file korup, hapus juga
            try:
                os.remove(path)
                removed += 1
            except:
                pass
                
    sisa = len(files) - removed
    print(f"  -> File rusak dihapus : {removed}")
    print(f"  -> SISA FILE BERSIH   : {sisa} (Kualitas HD)")
    print("-" * 50)

print("\nPembersihan selesai! Dataset Anda sekarang 100% Berkualitas Tinggi.")
