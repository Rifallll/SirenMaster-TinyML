import os
import re
import glob

DATASET_PATH = r"C:\Users\ASUS\Videos\DATASET"
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

def get_improved_group_name(file_path):
    base = os.path.splitext(os.path.basename(file_path))[0]
    # Strip aug prefix first!
    base = re.sub(r'^aug_(noise|pitch|stretch|speed|hardneg)_?\d*_?', '', base)
    match = re.match(r'^([a-zA-Z]+_\d+)', base)
    if match:
        return match.group(1)
    cleaned = base
    cleaned = re.sub(r'\s*\(\d+\)$', '', cleaned)
    cleaned = re.sub(r'_(original|loud|noise_light|shift|seg\d+|_seg\d+|part\d+)$', '', cleaned)
    cleaned = cleaned.replace('-[AudioTrimmer.com]', '').strip()
    return cleaned

for cat in CATEGORIES:
    cat_dir = os.path.join(DATASET_PATH, cat)
    files = glob.glob(os.path.join(cat_dir, "*.wav"))
    print(f"\nCategory: {cat}")
    count = 0
    for f in files:
        base = os.path.basename(f)
        if "aug_" in base:
            g = get_improved_group_name(f)
            # Find the corresponding original group name
            # e.g., if g is "aug_pitch_1234_ambulance_0113", the original might be "ambulance_0113"
            print(f"  File: {base} -> Group: {g}")
            count += 1
            if count >= 5:
                break
