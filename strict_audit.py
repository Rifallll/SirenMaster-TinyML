import os
import soundfile as sf
import numpy as np
from concurrent.futures import ThreadPoolExecutor

dataset_dir = r"c:\Users\ASUS\Videos\DATASET"
classes = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

print("=== MEMULAI AUDIT KETAT (STRICT AUDIT) ===")
print("Pengecekan: File corrupt, file kosong (0 bytes), dan file tanpa suara (silent).")

errors = []
filepaths = {}

def check_file(filepath):
    try:
        # Check size
        if os.path.getsize(filepath) == 0:
            return f"[KOSONG/0 BYTES] {filepath}"
            
        # Check audio readability
        info = sf.info(filepath)
        
        # We don't check duration here because short files are padded by the script anyway
        # But we will check for pure silence
        data, samplerate = sf.read(filepath)
        if len(data.shape) > 1:
            data = data.mean(axis=1) # to mono
        rms = np.sqrt(np.mean(data**2))
        if rms < 0.00001:
            return f"[SILENT/TIDAK ADA SUARA] {filepath}"
            
        return None
    except Exception as e:
        return f"[RUSAK/CORRUPT] {filepath}"

# Gather all files
all_files = []
for cls in classes:
    cls_dir = os.path.join(dataset_dir, cls)
    if os.path.exists(cls_dir):
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith('.wav')]
        all_files.extend(files)

print(f"Total file yang akan dicek fisiknya: {len(all_files)}")

with ThreadPoolExecutor(max_workers=8) as executor:
    results = executor.map(check_file, all_files)
    
for res in results:
    if res is not None:
        errors.append(res)

print("\n" + "="*50)
print("HASIL AUDIT KETAT FISIK FILE:")
print("="*50)

if len(errors) == 0:
    print("✨ LUAR BIASA! 100% FILE VALID & SEHAT.")
    print("Tidak ditemukan satupun file corrupt, tidak bisa dibuka, atau kosong!")
else:
    print(f"Ditemukan {len(errors)} file bermasalah (bug/rusak):")
    for e in errors[:50]:
        print(f" - {e}")
        
    print("\nMenghapus file yang bermasalah secara otomatis...")
    for f_err in errors:
        filepath = f_err.split("] ")[1]
        try:
            os.remove(filepath)
        except:
            pass
    print("Semua file rusak (bug) telah dibasmi dari dataset!")
