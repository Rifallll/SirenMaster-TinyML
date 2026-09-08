import os
import hashlib
from collections import defaultdict

def get_file_hash(filepath):
    # Menggunakan MD5 untuk kecepatan
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

hashes = defaultdict(list)
duplicate_count = 0
duplicate_files = []
cross_class_duplicates = []

print("Menganalisa file untuk mencari duplikat berdasarkan isi suaranya (MD5 Hash)...")
print("Ini akan memastikan apakah ada file yang dikopi paste berulang kali atau salah masuk kelas.")

for cls in classes:
    cls_dir = os.path.join(dataset_dir, cls)
    if not os.path.exists(cls_dir): continue
    
    files = os.listdir(cls_dir)
    print(f"Memproses {len(files)} file di folder {cls}...")
    for f in files:
        filepath = os.path.join(cls_dir, f)
        if os.path.isfile(filepath):
            file_hash = get_file_hash(filepath)
            if file_hash:
                hashes[file_hash].append((cls, filepath))

# Report duplicates
for file_hash, file_list in hashes.items():
    if len(file_list) > 1:
        duplicate_count += (len(file_list) - 1)
        duplicate_files.append(file_list)
        
        # Check if duplicates span across different classes (Very dangerous)
        classes_in_set = set([item[0] for item in file_list])
        if len(classes_in_set) > 1:
            cross_class_duplicates.append(file_list)

print(f"\n====================================")
print(f"HASIL AUDIT DUPLIKASI:")
print(f"Ditemukan {duplicate_count} file duplikat (isi audio persis sama 100%).")
print(f"Dari jumlah tersebut, ada {len(cross_class_duplicates)} set file yang diduplikat LINTAS KELAS (misal: satu suara ada di Ambulance dan Polisi sekaligus).")

if len(cross_class_duplicates) > 0:
    print("\n--- PERINGATAN: Duplikat Lintas Kelas (Sangat Berbahaya) ---")
    for dup_set in cross_class_duplicates[:5]:
        print("Set:")
        for cls, df in dup_set:
            print(f"  - [{cls}] {os.path.basename(df)}")

elif duplicate_count > 0:
    print("\n--- Contoh Duplikat di Kelas yang Sama (Aman, tapi bikin model overfitting) ---")
    for dup_set in duplicate_files[:5]:
        print("Set:")
        for cls, df in dup_set:
            print(f"  - [{cls}] {os.path.basename(df)}")

