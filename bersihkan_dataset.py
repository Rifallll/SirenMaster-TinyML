"""
BERSIHKAN DATASET — Hapus file bermasalah berdasarkan evaluation_results.json
Hanya menghapus file yang AI SANGAT YAKIN (>= 80%) tapi SALAH LABEL.
File-file ini adalah kandidat kuat data yang salah masuk folder.
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

EVAL_PATH = "evaluation_results.json"
# Threshold: hanya hapus jika confidence salah >= nilai ini
# Semakin tinggi = semakin selektif (aman)
CONFIDENCE_MIN = 0.80

print("=" * 60)
print("  PEMBERSIH DATASET — SIREN MASTER AI")
print("  Berdasarkan data nyata dari evaluation_results.json")
print("=" * 60)

with open(EVAL_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

false_examples = data.get("false_examples", [])
print(f"\n[INFO] Total sampel salah klasifikasi: {len(false_examples)}")
print(f"[INFO] Filter: hanya hapus jika yakin salah >= {CONFIDENCE_MIN*100:.0f}%\n")

# Kelompokkan berdasarkan true label dan predicted label
to_delete = []
skipped = []

for item in false_examples:
    filepath = item["filepath"]
    true_label = item["true"]
    pred_label = item["pred"]
    conf = item["confidence"]

    if conf >= CONFIDENCE_MIN:
        to_delete.append(item)
    else:
        skipped.append(item)

# Tampilkan ringkasan yang akan dihapus
print(f"{'='*60}")
print(f"  FILE YANG AKAN DIHAPUS (yakin salah >= {CONFIDENCE_MIN*100:.0f}%)")
print(f"{'='*60}")

categories = {}
for item in to_delete:
    key = f"{item['true']} → {item['pred']}"
    if key not in categories:
        categories[key] = []
    categories[key].append(item)

for key, items in sorted(categories.items()):
    print(f"\n  [{key}] — {len(items)} file:")
    for item in items:
        fname = os.path.basename(item['filepath'])
        print(f"    ❌ {fname:55s} (yakin: {item['confidence']*100:.1f}%)")

print(f"\n{'='*60}")
print(f"  RINGKASAN:")
print(f"  File akan dihapus : {len(to_delete)}")
print(f"  File dilewati     : {len(skipped)} (yakin salah < {CONFIDENCE_MIN*100:.0f}%)")
print(f"{'='*60}")

# Konfirmasi
print("\n⚠️  File-file di atas akan DIHAPUS PERMANEN dari dataset.")
print("    Ketik 'YA' untuk lanjut, atau tekan Enter untuk batal: ", end="")
konfirmasi = input().strip().upper()

if konfirmasi != "YA":
    print("\n[BATAL] Tidak ada file yang dihapus.")
    sys.exit(0)

# Hapus file
print("\n[*] Mulai menghapus...")
berhasil = 0
gagal = 0
tidak_ada = 0

for item in to_delete:
    filepath = item["filepath"]
    fname = os.path.basename(filepath)
    
    if not os.path.exists(filepath):
        print(f"  [SKIP] Tidak ada: {fname}")
        tidak_ada += 1
        continue
    
    try:
        os.remove(filepath)
        print(f"  [OK] Hapus: {fname}")
        berhasil += 1
    except Exception as e:
        print(f"  [ERR] Gagal hapus {fname}: {e}")
        gagal += 1

print(f"\n{'='*60}")
print(f"  SELESAI!")
print(f"  Berhasil dihapus : {berhasil} file")
print(f"  Tidak ditemukan  : {tidak_ada} file (sudah terhapus sebelumnya)")
print(f"  Gagal dihapus    : {gagal} file")
print(f"{'='*60}")

if berhasil > 0:
    print(f"\n✅ Dataset sudah lebih bersih!")
    print(f"   Jalankan training ulang:")
    print(f"   python train_lokal.py")
