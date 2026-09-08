import os
import re

dataset_dir = r"c:\Users\ASUS\Videos\DATASET"
classes = ["AMBULANCE", "FIRETRUCK", "NORMAL", "POLICE"]

# Keywords associated with each class (strict cross-contamination check)
keywords = {
    "AMBULANCE": ["ambulance", "ambulans", "amb_"],
    "FIRETRUCK": ["firetruck", "damkar", "pemadam", "fire_"],
    "POLICE": ["police", "polisi", "pol_"],
    "NORMAL": ["normal_"]
}

# The word 'noise' is part of augmentation (aug_noise_...), so we don't flag it as a cross-class keyword.

misplaced_files = []

for cls in classes:
    cls_dir = os.path.join(dataset_dir, cls)
    if not os.path.exists(cls_dir): continue
    
    # Tentukan keyword apa saja yang TIDAK BOLEH ada di folder ini
    forbidden_keywords = []
    for other_cls, kw_list in keywords.items():
        if other_cls != cls:
            forbidden_keywords.extend(kw_list)
            
    # Also for NORMAL, maybe it shouldn't contain 'siren'
    if cls == "NORMAL":
        forbidden_keywords.append("siren")
            
    files = os.listdir(cls_dir)
    for f in files:
        f_lower = f.lower()
        
        # Remove common prefix like 'aug_noise_' 'aug_pitch_' 'aug_speed_' to avoid false matches
        f_clean = re.sub(r'aug_(noise|pitch|speed)_?\d*_?', '', f_lower)

        for kw in forbidden_keywords:
            if kw in f_clean:
                misplaced_files.append((cls, f, kw))
                break 

print(f"\nDitemukan {len(misplaced_files)} file yang berpotensi salah kamar berdasarkan namanya.")
if misplaced_files:
    print("Contoh file yang mencurigakan:")
    for cls, f, kw in misplaced_files[:50]:
        print(f"  - Di folder [{cls}] ada file '{f}' (Mengandung kata '{kw}')")
else:
    print("Semua nama file terlihat aman dan tidak ada indikasi tertukar secara nama file.")
