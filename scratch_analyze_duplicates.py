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
    print(f"Hashing {total_files} files in parallel...")
    
    file_hashes = {}
    cross_class = []
    same_class = []
    
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
                        orig_fp, orig_cat = file_hashes[fhash]
                        if orig_cat != cat:
                            cross_class.append((fp, cat, orig_fp, orig_cat))
                        else:
                            same_class.append((fp, orig_fp))
                    else:
                        file_hashes[fhash] = (fp, cat)
            except Exception as e:
                pass
                
    print(f"\nTotal duplicates found: {len(cross_class) + len(same_class)}")
    print(f"Same-class duplicates: {len(same_class)}")
    print(f"Cross-class duplicates: {len(cross_class)}")
    
    if cross_class:
        print("\nCross-class duplicates found (Very Bad!):")
        for fp, cat, orig_fp, orig_cat in cross_class[:20]:
            print(f"  - [{cat}] {os.path.basename(fp)} <---> [{orig_cat}] {os.path.basename(orig_fp)}")

if __name__ == '__main__':
    main()
