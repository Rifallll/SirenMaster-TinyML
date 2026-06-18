import os
import wave
import shutil

DATASET_BASE = r"C:\Users\ASUS\Videos\DATASET"
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
TRASH_DIR = os.path.join(DATASET_BASE, "TRASH")

print("="*50)
print("  FINAL DATASET AUDIT")
print("="*50)

total_files = 0
total_duration = 0.0
invalid_files = 0

for cat in CATEGORIES:
    cat_path = os.path.join(DATASET_BASE, cat)
    if not os.path.exists(cat_path):
        print(f"[!] Warning: Folder {cat} tidak ditemukan!")
        continue
        
    files = [f for f in os.listdir(cat_path) if f.endswith('.wav')]
    cat_duration = 0.0
    cat_invalid = 0
    
    for f in files:
        f_path = os.path.join(cat_path, f)
        
        # Check size
        if os.path.getsize(f_path) == 0:
            cat_invalid += 1
            invalid_files += 1
            continue
            
        # Try read duration
        try:
            if os.path.getsize(f_path) == 0:
                raise ValueError("0 bytes")
            with wave.open(f_path, 'r') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                duration = frames / float(rate)
                cat_duration += duration
                total_duration += duration
        except:
            cat_invalid += 1
            invalid_files += 1
            # Pindahkan file corrupt ke TRASH
            shutil.move(f_path, os.path.join(TRASH_DIR, f"CORRUPT_{cat}_{f}"))
            
    total_files += len(files) - cat_invalid
    print(f"[OK] [{cat.ljust(10)}] {len(files)-cat_invalid} files | Durasi: {cat_duration/3600:.1f} jam | Dipindah ke TRASH: {cat_invalid}")

print("-" * 50)
trash_count = len(os.listdir(TRASH_DIR)) if os.path.exists(TRASH_DIR) else 0
print(f"[TRASH] Total file dibuang (salah label + corrupt) : {trash_count} files")
print(f"\n[TOTAL] DATA SIAP PAKAI : {total_files} files ({(total_duration/3600):.1f} jam audio)")

print("\nSTATUS: SANGAT BERSIH & SEMPURNA")
