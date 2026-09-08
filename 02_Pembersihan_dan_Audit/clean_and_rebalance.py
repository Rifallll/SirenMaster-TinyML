import os
import hashlib
import random

random.seed(42)

def get_file_hash(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        return None

dataset_dir = r"c:\Users\ASUS\Videos\DATASET"
classes = ["AMBULANCE", "FIRETRUCK", "NORMAL", "POLICE"]

# Step 1: Remove exact duplicates
print("--- TAHAP 1: Menghapus File Duplikat (100% Bersih) ---")
class_counts = {}

for cls in classes:
    cls_dir = os.path.join(dataset_dir, cls)
    if not os.path.exists(cls_dir): continue
    
    hashes = set()
    files = os.listdir(cls_dir)
    removed_count = 0
    
    for f in files:
        filepath = os.path.join(cls_dir, f)
        if os.path.isfile(filepath):
            file_hash = get_file_hash(filepath)
            if file_hash:
                if file_hash in hashes:
                    os.remove(filepath)
                    removed_count += 1
                else:
                    hashes.add(file_hash)
    
    new_count = len(os.listdir(cls_dir))
    class_counts[cls] = new_count
    print(f"[{cls}] Berhasil menghapus {removed_count} duplikat. Sisa file unik: {new_count}")

# Step 2: Find minimum count
min_count = min(class_counts.values())
print(f"\n--- TAHAP 2: Meratakan Ulang (Re-balance) ke {min_count} File ---")

for cls in classes:
    cls_dir = os.path.join(dataset_dir, cls)
    if not os.path.exists(cls_dir): continue
    
    current_count = len(os.listdir(cls_dir))
    
    if current_count > min_count:
        excess = current_count - min_count
        files = os.listdir(cls_dir)
        files_to_remove = random.sample(files, excess)
        
        for f in files_to_remove:
            os.remove(os.path.join(cls_dir, f))
            
        print(f"[{cls}] Dipotong acak {excess} file. Sekarang persis: {min_count}")
    else:
        print(f"[{cls}] Sudah di batas rata: {min_count}")

print(f"\n✅ SELESAI! Dataset sekarang 100% UNIK dan SEIMBANG SEMPURNA ({min_count} file per kelas).")
