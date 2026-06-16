import os
import hashlib
import numpy as np
import soundfile as sf

DATASET_PATH = r"C:\Users\ASUS\Videos\DATASET"
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'POLICE', 'NORMAL']

def get_file_hash(file_path):
    """Calculate MD5 checksum of a file to check for exact duplicates."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def analyze_dataset():
    print("="*60)
    print("           DATASET DIAGNOSTICS & AUDIT REPORT")
    print("="*60)
    
    corrupted_files = []
    silent_files = []
    short_files = [] # Less than 0.5 seconds
    file_hashes = {}
    duplicates = []
    
    stats = {cat: {"total": 0, "corrupted": 0, "silent": 0, "short": 0, "valid": 0} for cat in CATEGORIES}
    
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATASET_PATH, cat)
        if not os.path.exists(cat_dir):
            print(f"[!] Warning: Folder '{cat}' does not exist!")
            continue
            
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if not file.lower().endswith('.wav'):
                    continue
                    
                file_path = os.path.join(root, file)
                stats[cat]["total"] += 1
                
                # Check for exact duplicate files by hashing
                try:
                    fhash = get_file_hash(file_path)
                    if fhash in file_hashes:
                        duplicates.append((file_path, file_hashes[fhash]))
                    else:
                        file_hashes[fhash] = file_path
                except Exception as e:
                    pass
                
                # Try loading the audio file
                try:
                    y, sr = sf.read(file_path)
                    
                    # Convert to mono if stereo
                    if len(y.shape) > 1:
                        y = np.mean(y, axis=1)
                        
                    duration = len(y) / sr
                    
                    # Check if file is silent (RMS amplitude extremely low)
                    rms = np.sqrt(np.mean(y**2))
                    if rms < 0.0005:
                        silent_files.append((file_path, rms))
                        stats[cat]["silent"] += 1
                    # Check if file is extremely short
                    elif duration < 0.5:
                        short_files.append((file_path, duration))
                        stats[cat]["short"] += 1
                    else:
                        stats[cat]["valid"] += 1
                        
                except Exception as e:
                    corrupted_files.append((file_path, str(e)))
                    stats[cat]["corrupted"] += 1
                    
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
    
    # Save a detailed log of defects
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
            
    print("[SUCCESS] Diagnostics complete. Defects logged to 'dataset_defects_log.txt'.")

if __name__ == '__main__':
    analyze_dataset()
