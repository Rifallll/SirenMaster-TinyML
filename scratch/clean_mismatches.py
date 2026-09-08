import os
import re
import shutil

DATASET_DIR = r"C:\Users\ASUS\Videos\DATASET"
REPORT_PATH = os.path.join(DATASET_DIR, "audit_total_mismatches.txt")
TRASH_DIR = os.path.join(DATASET_DIR, "TRASH")

def clean_mismatches():
    print("="*60)
    print("        MEMBERSIHKAN FILE SALAH FOLDER KE FOLDER TRASH")
    print("="*60)
    
    if not os.path.exists(REPORT_PATH):
        print(f"[ERROR] File laporan '{REPORT_PATH}' tidak ditemukan! Jalankan audit_total.py terlebih dahulu.")
        return

    if not os.path.exists(TRASH_DIR):
        os.makedirs(TRASH_DIR)
        print(f"[INFO] Membuat folder karantina: {TRASH_DIR}")

    # Read mismatch files path from report
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all path blocks
    paths = re.findall(r'Path:\s*(.+)', content)
    
    if len(paths) == 0:
        print("[SUCCESS] Tidak ada file mencurigakan yang terdaftar untuk dibersihkan.")
        return

    print(f"[INFO] Ditemukan {len(paths)} file yang akan dipindahkan ke folder TRASH.")

    moved_count = 0
    for path in paths:
        path = path.strip()
        if os.path.exists(path):
            file_name = os.path.basename(path)
            dest_path = os.path.join(TRASH_DIR, file_name)
            
            # If target file already exists in TRASH, generate unique name
            base, ext = os.path.splitext(file_name)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(TRASH_DIR, f"{base}_moved_{counter}{ext}")
                counter += 1
                
            try:
                shutil.move(path, dest_path)
                print(f"  [OK] Moved: {file_name} -> TRASH")
                moved_count += 1
            except Exception as e:
                print(f"  [FAILED] Gagal memindahkan {file_name}: {e}")
        else:
            print(f"  [WARN] File tidak ditemukan di sistem: {path}")

    print(f"\n[SUCCESS] Berhasil memindahkan {moved_count} file ke folder TRASH.")
    
    # Hapus cache agar training berikutnya memproses data yang sudah bersih
    cache_path = os.path.join(DATASET_DIR, "siren_40x249_v2_clean_cache.npz")
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print("[INFO] File cache 'siren_40x249_v2_clean_cache.npz' berhasil dihapus.")
        
    print("="*60)

if __name__ == "__main__":
    clean_mismatches()
