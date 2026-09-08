import os
import hashlib
import numpy as np
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed

DATASET_PATH = r"C:\Users\ASUS\Videos\DATASET"
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'POLICE', 'NORMAL']

def get_file_hash_and_audio_stats(file_path):
    """Worker function to hash and check audio properties of a single file."""
    try:
        # Check size first
        sz = os.path.getsize(file_path)
        if sz == 0:
            return file_path, "corrupted", "0 bytes file", None
            
        # Hash check
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        fhash = hasher.hexdigest()
        
        # Audio check
        y, sr = sf.read(file_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1) # mono
            
        duration = len(y) / sr
        rms = np.sqrt(np.mean(y**2))
        
        if rms < 0.0005:
            return file_path, "silent", rms, fhash
        elif duration < 0.5:
            return file_path, "short", duration, fhash
        else:
            return file_path, "valid", rms, fhash
            
    except Exception as e:
        return file_path, "corrupted", str(e), None

def main():
    print("="*60)
    print("      FAST PARALLEL DATASET DIAGNOSTICS & AUDIT")
    print("="*60)
    
    file_list = []
    file_categories = {}
    
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATASET_PATH, cat)
        if not os.path.exists(cat_dir):
            print(f"[!] Warning: Folder '{cat}' does not exist!")
            continue
            
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    fp = os.path.join(root, file)
                    file_list.append(fp)
                    file_categories[fp] = cat
                    
    total_files = len(file_list)
    print(f"Scanning {total_files} files in parallel...")
    
    corrupted_files = []
    silent_files = []
    short_files = []
    valid_files = []
    file_hashes = {}
    duplicates = []
    
    stats = {cat: {"total": 0, "corrupted": 0, "silent": 0, "short": 0, "valid": 0} for cat in CATEGORIES}
    
    # Process files in parallel
    max_workers = min(12, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_file_hash_and_audio_stats, fp): fp for fp in file_list}
        
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            if completed % 1000 == 0 or completed == total_files:
                print(f"  Processed {completed}/{total_files} files...")
                
            fp = futures[fut]
            cat = file_categories[fp]
            stats[cat]["total"] += 1
            
            try:
                fp, status, val, fhash = fut.result()
                
                # Check duplicate
                if fhash:
                    if fhash in file_hashes:
                        duplicates.append((fp, file_hashes[fhash]))
                    else:
                        file_hashes[fhash] = fp
                        
                if status == "valid":
                    stats[cat]["valid"] += 1
                    valid_files.append(fp)
                elif status == "silent":
                    stats[cat]["silent"] += 1
                    silent_files.append((fp, val))
                elif status == "short":
                    stats[cat]["short"] += 1
                    short_files.append((fp, val))
                elif status == "corrupted":
                    stats[cat]["corrupted"] += 1
                    corrupted_files.append((fp, val))
            except Exception as e:
                stats[cat]["corrupted"] += 1
                corrupted_files.append((fp, str(e)))
                
    print("\n[STATS] Class Distribution and Status:")
    for cat in CATEGORIES:
        c_stats = stats[cat]
        print(f"  {cat:<12}: Total={c_stats['total']:<5} | Valid={c_stats['valid']:<5} | Corrupted={c_stats['corrupted']:<3} | Silent={c_stats['silent']:<3} | Too Short={c_stats['short']:<3}")
        
    print("\n[DEFECTS] Summary of Deficiencies Found:")
    print(f"  - Corrupted WAV files (cannot be read): {len(corrupted_files)}")
    print(f"  - Silent WAV files (near-zero amplitude): {len(silent_files)}")
    print(f"  - Too short WAV files (< 0.5s): {len(short_files)}")
    print(f"  - Duplicate files (identical content hashes): {len(duplicates)}")
    
    if len(corrupted_files) > 0:
        print("\n[WARN] Sample Corrupted Files (first 5):")
        for fp, err in corrupted_files[:5]:
            print(f"    - {os.path.relpath(fp, DATASET_PATH)}: {err}")
            
    if len(silent_files) > 0:
        print("\n[WARN] Sample Silent Files (first 5):")
        for fp, rms in silent_files[:5]:
            print(f"    - {os.path.relpath(fp, DATASET_PATH)} (RMS={rms:.6f})")
            
    if len(duplicates) > 0:
        print("\n[WARN] Sample Duplicate Files (first 5):")
        for fp, orig in duplicates[:5]:
            print(f"    - Duplicate: {os.path.relpath(fp, DATASET_PATH)}")
            print(f"      Original:  {os.path.relpath(orig, DATASET_PATH)}")
            
    print("="*60)
    
    # Save log
    with open(os.path.join(DATASET_PATH, "dataset_defects_log.txt"), "w") as log:
        log.write("=== DATASET DEFECTS LOG ===\n\n")
        log.write(f"Corrupted Files Count: {len(corrupted_files)}\n")
        for fp, err in corrupted_files:
            log.write(f"CORRUPTED: {fp} | Error: {err}\n")
        log.write(f"\nSilent Files Count: {len(silent_files)}\n")
        for fp, rms in silent_files:
            log.write(f"SILENT: {fp} | RMS: {rms}\n")
        log.write(f"\nDuplicate Files Count: {len(duplicates)}\n")
        for fp, orig in duplicates:
            log.write(f"DUPLICATE: {fp} | Original: {orig}\n")
            
    print("[SUCCESS] Fast diagnostics complete. Logged to 'dataset_defects_log.txt'.")

if __name__ == '__main__':
    main()
