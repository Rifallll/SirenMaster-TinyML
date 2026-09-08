import os
import re
import soundfile as sf
import numpy as np
from collections import Counter, defaultdict

DATASET_PATH = r"C:\Users\ASUS\Videos\DATASET"
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'POLICE', 'NORMAL']

def get_improved_group_name(file_path):
    base = os.path.splitext(os.path.basename(file_path))[0]
    match = re.match(r'^([a-zA-Z]+_\d+)', base)
    if match:
        return match.group(1)
    cleaned = base
    cleaned = re.sub(r'\s*\(\d+\)$', '', cleaned)
    cleaned = re.sub(r'_(original|loud|noise_light|shift|seg\d+|_seg\d+)$', '', cleaned)
    cleaned = cleaned.replace('-[AudioTrimmer.com]', '').strip()
    return cleaned

def simulate_chunking():
    print("="*60)
    print("CHUNK SIMULATION DIAGNOSTICS - STRATIFIED LIMITS")
    print("="*60)
    
    CHUNK_DUR = 2.0
    
    # We will define class-specific extraction settings
    # To naturally balance classes:
    # - AMBULANCE: Overlapping chunks (50% overlap, hop=1.0s), max 10 chunks per file
    # - POLICE: Non-overlapping chunks, max 2 chunks per file
    # - NORMAL: Non-overlapping chunks, max 2 chunks per file
    # - FIRETRUCK: Non-overlapping chunks, max 1 chunk per file
    
    class_raw_counts = Counter()
    class_chunk_counts = Counter()
    
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATASET_PATH, cat)
        if not os.path.exists(cat_dir):
            continue
            
        # Class-specific config
        if cat == 'AMBULANCE':
            hop_dur = 1.0  # 50% overlap
            max_chunks = 8
        elif cat == 'POLICE':
            hop_dur = 2.0  # no overlap
            max_chunks = 2
        elif cat == 'NORMAL':
            hop_dur = 2.0  # no overlap
            max_chunks = 2
        else: # FIRETRUCK
            hop_dur = 2.0  # no overlap
            max_chunks = 1
            
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if not file.lower().endswith('.wav'):
                    continue
                file_path = os.path.join(root, file)
                class_raw_counts[cat] += 1
                
                try:
                    info = sf.info(file_path)
                    duration = info.duration
                    
                    if duration < CHUNK_DUR:
                        if duration >= 0.5:
                            num_chunks = 1
                        else:
                            num_chunks = 0
                    else:
                        num_chunks = int((duration - CHUNK_DUR) // hop_dur) + 1
                        
                    num_chunks = min(num_chunks, max_chunks)
                    class_chunk_counts[cat] += num_chunks
                except Exception:
                    pass
                    
    print("Raw file distribution:")
    for cat in CATEGORIES:
        print(f"  {cat:<12}: {class_raw_counts[cat]}")
        
    print("\nBalanced chunk distribution:")
    for cat in CATEGORIES:
        print(f"  {cat:<12}: {class_chunk_counts[cat]} chunks")
        
    print("="*60)

if __name__ == '__main__':
    simulate_chunking()
