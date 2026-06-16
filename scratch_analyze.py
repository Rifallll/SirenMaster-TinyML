import os
import re
from collections import defaultdict

DATASET_PATH = r"C:\Users\ASUS\Videos\DATASET"
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'POLICE', 'NORMAL']

def get_current_group_name(file_path):
    base = os.path.splitext(os.path.basename(file_path))[0]
    match = re.match(r'^([a-zA-Z]+_\d+)', base)
    if match:
        return match.group(1)
    return base

def get_improved_group_name(file_path):
    base = os.path.splitext(os.path.basename(file_path))[0]
    # Check standard class_num pattern first (e.g. fire_0001)
    match = re.match(r'^([a-zA-Z]+_\d+)', base)
    if match:
        return match.group(1)
        
    cleaned = base
    # Remove trailing numbers in parentheses like (1), (2)
    cleaned = re.sub(r'\s*\(\d+\)$', '', cleaned)
    # Remove common audio trim/augment suffixes
    cleaned = re.sub(r'_(original|loud|noise_light|shift|seg\d+|_seg\d+)$', '', cleaned)
    # Remove -[AudioTrimmer.com]
    cleaned = cleaned.replace('-[AudioTrimmer.com]', '')
    # Remove trailing spaces
    cleaned = cleaned.strip()
    return cleaned

def main():
    print("="*60)
    print("ANALYSIS OF DATASET GROUPS AND LEAKAGE")
    print("="*60)
    
    current_groups = defaultdict(list)
    improved_groups = defaultdict(list)
    
    total_files = 0
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATASET_PATH, cat)
        if not os.path.exists(cat_dir):
            continue
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if not file.lower().endswith('.wav'):
                    continue
                total_files += 1
                fp = os.path.join(root, file)
                
                curr_g = get_current_group_name(fp)
                current_groups[curr_g].append((cat, file))
                
                imp_g = get_improved_group_name(fp)
                improved_groups[imp_g].append((cat, file))
                
    print(f"Total files analyzed: {total_files}")
    print(f"Current grouping has {len(current_groups)} unique groups.")
    print(f"Improved grouping has {len(improved_groups)} unique groups.")
    
    # We want to check for potential leakage in current grouping
    # Leakage happens when files that belong to the SAME improved group (same original recording)
    # get split into DIFFERENT current groups, which allows them to end up in both Train and Test.
    
    leakage_count = 0
    leakage_groups = []
    
    for imp_g, files in improved_groups.items():
        curr_g_set = set(get_current_group_name(f[1]) for f in files)
        if len(curr_g_set) > 1:
            leakage_count += 1
            leakage_groups.append((imp_g, curr_g_set, files))
            
    print(f"\n[LEAKAGE REPORT]")
    print(f"Number of root recordings split across multiple groups (direct leakage): {leakage_count}")
    
    if leakage_count > 0:
        print("\nExamples of leakage (showing how one recording is split into different group names):")
        for imp_g, curr_g_set, files in leakage_groups[:5]:
            print(f"\nRoot recording base: '{imp_g}'")
            print(f"  Split into {len(curr_g_set)} groups: {curr_g_set}")
            print(f"  Total files affected: {len(files)}")
            for cat, file in files[:6]:
                print(f"    - [{cat}] {file} (Current Group: '{get_current_group_name(file)}')")
                
    print("="*60)

if __name__ == '__main__':
    main()
