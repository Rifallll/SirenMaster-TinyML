import os
import glob
import csv

ROOT = r"C:\Users\ASUS\Videos\DATASET"
TRASH_DIR = os.path.join(ROOT, "TRASH", "audit_hasil_pemisahan")
CSV_PATH = os.path.join(ROOT, "daftar_file_anomali.csv")

bukan_sirine_files = glob.glob(os.path.join(TRASH_DIR, "bukan_sirine", "*.*"))
salah_kamar_files = glob.glob(os.path.join(TRASH_DIR, "salah_kamar", "*.*"))
hening_files = glob.glob(os.path.join(TRASH_DIR, "hening_rusak", "*.*"))

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["No", "Nama File", "Jenis Anomali", "Folder Asal", "Prediksi AI", "Lokasi Penyimpanan File"])
    
    no = 1
    for fp in bukan_sirine_files:
        fname = os.path.basename(fp)
        orig = fname.split('_')[0] if '_' in fname else 'UNKNOWN'
        writer.writerow([no, fname, "Bukan Sirine (Noise Murni)", orig, "NORMAL", fp])
        no += 1
        
    for fp in salah_kamar_files:
        fname = os.path.basename(fp)
        parts = fname.split('_')
        orig = parts[1] if len(parts) > 1 else 'UNKNOWN'
        target = parts[3] if len(parts) > 3 else 'UNKNOWN'
        writer.writerow([no, fname, "Salah Kamar (Tertukar)", orig, target, fp])
        no += 1

    for fp in hening_files:
        fname = os.path.basename(fp)
        writer.writerow([no, fname, "Hening / Corrupt", "UNKNOWN", "NONE", fp])
        no += 1

print(f"[+] File CSV daftar anomali berhasil dibuat: {CSV_PATH}")
