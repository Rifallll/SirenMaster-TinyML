import os

CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
DATASET_PATHS = [r"C:\Users\ASUS\Videos\DATASET"]

file_paths = []
for idx, cat in enumerate(CATEGORIES):
    cat_files = []
    for base_path in DATASET_PATHS:
        cat_dir = os.path.join(base_path, cat)
        if not os.path.exists(cat_dir):
            continue
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    cat_files.append(os.path.join(root, file))
    print(f"Category '{cat}': found {len(cat_files)} files.")
    file_paths.extend(cat_files)
print(f"Total files: {len(file_paths)}")
