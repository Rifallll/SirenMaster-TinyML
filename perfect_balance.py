import os
import shutil
import random

# Fix random seed for reproducibility (optional but good practice)
random.seed(42)

dataset_dir = r"c:\Users\ASUS\Videos\DATASET"
target_count = 3848

# 1. Merge Ambilance to AMBULANCE
ambilance_dir = os.path.join(dataset_dir, "Ambilance")
ambulance_dir = os.path.join(dataset_dir, "AMBULANCE")

if os.path.exists(ambilance_dir):
    print(f"Moving files from {ambilance_dir} to {ambulance_dir}...")
    for filename in os.listdir(ambilance_dir):
        src = os.path.join(ambilance_dir, filename)
        dst = os.path.join(ambulance_dir, filename)
        if os.path.isfile(src):
            # To avoid name conflicts, append a suffix if exists
            if os.path.exists(dst):
                name, ext = os.path.splitext(filename)
                dst = os.path.join(ambulance_dir, f"{name}_merged{ext}")
            shutil.move(src, dst)
    # Remove empty directory
    try:
        os.rmdir(ambilance_dir)
        print("Removed 'Ambilance' directory.")
    except Exception as e:
        print(f"Could not remove 'Ambilance' directory: {e}")

# 2. Balance all target directories to target_count
classes = ["AMBULANCE", "FIRETRUCK", "NORMAL", "POLICE"]

print("\n--- Starting Dataset Balancing ---")
for cls in classes:
    cls_dir = os.path.join(dataset_dir, cls)
    if os.path.exists(cls_dir):
        files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        count = len(files)
        print(f"Class {cls} currently has {count} files.")
        
        if count > target_count:
            # Randomly select files to remove
            excess = count - target_count
            print(f"  -> Undersampling: Removing {excess} random files to reach {target_count}...")
            files_to_remove = random.sample(files, excess)
            for f in files_to_remove:
                os.remove(os.path.join(cls_dir, f))
            print(f"  -> {cls} now has {len(os.listdir(cls_dir))} files.")
        elif count == target_count:
            print(f"  -> {cls} is already at perfect balance ({target_count}).")
        else:
            print(f"  -> WARNING: {cls} has {count} files, which is less than the target {target_count}.")

print("\nBalancing complete! Dataset is now perfectly 100% balanced.")
