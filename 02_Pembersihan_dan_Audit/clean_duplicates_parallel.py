import os
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

DATASET_PATH = r"C:\Users\ASUS\Videos\DATASET"
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'POLICE', 'NORMAL']

def hash_file(file_path):
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return file_path, hasher.hexdigest()
    except Exception as e:
        return file_path, None

def main():
    print("="*60)
    print("      PARALLEL DATASET DEDUPLICATION")
    print("="*60)
    
    file_list = []
    file_categories = {}
    
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATASET_PATH, cat)
        if not os.path.exists(cat_dir): continue
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    fp = os.path.join(root, file)
                    file_list.append(fp)
                    file_categories[fp] = cat
                    
    total_files = len(file_list)
    print(f"Scanning {total_files} files...")
    
    file_hashes = {}
    duplicates = []
    
    max_workers = min(12, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(hash_file, fp): fp for fp in file_list}
        
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            if completed % 2000 == 0 or completed == total_files:
                print(f"  Hashed {completed}/{total_files} files...")
                
            fp = futures[fut]
            cat = file_categories[fp]
            
            try:
                fp, fhash = fut.result()
                if fhash:
                    if fhash in file_hashes:
                        # Duplicate found!
                        duplicates.append((fp, file_hashes[fhash]))
                    else:
                        file_hashes[fhash] = fp
            except Exception as e:
                pass
                
    print(f"\nFound {len(duplicates)} duplicate files.")
    
    if duplicates:
        print("Deleting duplicate files...")
        deleted_count = 0
        for fp, orig in duplicates:
            try:
                os.remove(fp)
                deleted_count += 1
            except Exception as e:
                print(f"[!] Failed to delete {fp}: {e}")
        print(f"Successfully deleted {deleted_count} duplicate files.")
        
    print("\nUpdated counts per class:")
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATASET_PATH, cat)
        if os.path.exists(cat_dir):
            count = len([f for f in os.listdir(cat_dir) if f.lower().endswith('.wav')])
            print(f"  {cat:<12}: {count} files")
            
    print("="*60)
    print("[SUCCESS] Dataset deduplication completed successfully!")

if __name__ == '__main__':
    main()
