"""
HAPUS MASSAL — Berdasarkan hasil eval_terbaru.json
Hapus semua file yang AI yakin >= 90% tapi salah klasifikasi
"""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATASET_BASE = r"C:\Users\ASUS\Videos\DATASET"
EVAL_PATH    = os.path.join(DATASET_BASE, "eval_terbaru.json")
CONFIDENCE_MIN = 0.90  # Hanya hapus jika yakin salah >= 90%

with open(EVAL_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

errors = data.get("errors_high_conf", [])

# Filter sesuai threshold
to_delete = [e for e in errors if e['conf'] >= CONFIDENCE_MIN]

print("=" * 65)
print("  HAPUS MASSAL — FILE SALAH KLASIFIKASI YAKIN >= 90%")
print("=" * 65)

# Kelompokkan berdasarkan arah kesalahan
by_confusion = {}
for e in to_delete:
    key = f"{e['true']} → {e['pred']}"
    by_confusion.setdefault(key, []).append(e)

total = 0
for key, items in sorted(by_confusion.items()):
    print(f"\n  [{key}] — {len(items)} file")
    total += len(items)

print(f"\n{'='*65}")
print(f"  TOTAL FILE YANG AKAN DIHAPUS: {total}")
print(f"{'='*65}")
print(f"\n⚠️  Ketik 'YA' untuk hapus semua, atau Enter untuk batal: ", end="")

konfirmasi = input().strip().upper()
if konfirmasi != "YA":
    print("[BATAL]")
    sys.exit(0)

print("\n[*] Mulai menghapus...")
berhasil = gagal = tidak_ada = 0

for e in to_delete:
    folder   = e['folder']   # nama kategori, e.g. 'FIRETRUCK'
    fname    = e['file']
    filepath = os.path.join(DATASET_BASE, folder, fname)

    if not os.path.exists(filepath):
        tidak_ada += 1
        continue
    try:
        os.remove(filepath)
        berhasil += 1
        if berhasil % 100 == 0:
            print(f"  ... {berhasil} file dihapus", flush=True)
    except Exception as ex:
        print(f"  [ERR] {fname}: {ex}")
        gagal += 1

print(f"\n{'='*65}")
print(f"  SELESAI!")
print(f"  Berhasil dihapus : {berhasil}")
print(f"  Tidak ditemukan  : {tidak_ada}")
print(f"  Gagal            : {gagal}")
print(f"{'='*65}")

# Cek sisa dataset
print("\n[*] Sisa dataset setelah pembersihan:")
cats = ['AMBULANCE','FIRETRUCK','NORMAL','POLICE']
for c in cats:
    folder = os.path.join(DATASET_BASE, c)
    n = len([f for f in os.listdir(folder)
             if f.lower().endswith(('.wav','.m4a'))])
    print(f"  {c:12s}: {n} file")

print("\n✅ Jalankan training ulang:")
print("   python train_lokal.py")
