import os
import hashlib

DATASET_PATH = r"C:\Users\ASUS\Videos\DATASET"
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'POLICE', 'NORMAL']

def main():
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATASET_PATH, cat)
        if not os.path.exists(cat_dir): continue
        
        file_list = []
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    file_list.append(os.path.join(root, file))
                    
        hashes = set()
        duplicates = 0
        for fp in file_list:
            hasher = hashlib.md5()
            with open(fp, 'rb') as f:
                buf = f.read(65536)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(65536)
            fhash = hasher.hexdigest()
            if fhash in hashes:
                duplicates += 1
            else:
                hashes.add(fhash)
                
        print(f"[{cat}] Total: {len(file_list)} | Duplicates: {duplicates} | Unique: {len(file_list) - duplicates}")

if __name__ == '__main__':
    main()
