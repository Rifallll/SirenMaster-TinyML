import json
import os
import shutil

DATASET_BASE = r"C:\Users\ASUS\Videos\DATASET"
EVAL_PATH = os.path.join(DATASET_BASE, "eval_terbaru.json")
TRASH_DIR = os.path.join(DATASET_BASE, "TRASH")

if not os.path.exists(TRASH_DIR):
    os.makedirs(TRASH_DIR)

if os.path.exists(EVAL_PATH):
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    errors = data.get("errors_high_conf", [])
    
    # Hanya hapus file original (jangan file aug_ karena itu hasil copy script)
    non_aug = [e for e in errors if not e["file"].startswith("aug_")]
    
    moved_count = 0
    for e in non_aug:
        # e['file'] itu cuma nama filenya. Kita harus tau foldernya
        true_label = e['true']
        filename = e['file']
        
        src_path = os.path.join(DATASET_BASE, true_label, filename)
        dst_path = os.path.join(TRASH_DIR, f"{true_label}_{filename}")
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            moved_count += 1
            print(f"[MOVED] {true_label}/{filename} -> TRASH")
            
    print(f"\n[SUCCESS] Berhasil membuang {moved_count} file rusak/salah label ke folder TRASH.")
else:
    print(f"[ERROR] {EVAL_PATH} tidak ditemukan.")
