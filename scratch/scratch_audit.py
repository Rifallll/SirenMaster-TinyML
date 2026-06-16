import json
import os

filepath = "eval_terbaru.json"
if not os.path.exists(filepath):
    print("File eval_terbaru.json tidak ditemukan!")
    exit(1)

with open(filepath, "r", encoding="utf-8") as f:
    data = json.load(f)

print("============================================================")
print("             HASIL AUDIT EVALUASI DATASET & AI              ")
print("============================================================")

counts = data.get("counts", {})
total_benar = 0
total_all = 0

print("Akurasi Deteksi Per Kategori:")
for cat in ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']:
    if cat in counts:
        benar = counts[cat].get("benar", 0)
        total = counts[cat].get("total", 0)
        total_benar += benar
        total_all += total
        pct = (benar / total * 100) if total > 0 else 0
        print(f"  - {cat:<10} : {benar:4d} / {total:4d} benar ({pct:.2f}%)")

print("-" * 60)
overall_pct = (total_benar / total_all * 100) if total_all > 0 else 0
print(f"  TOTAL UTAMA : {total_benar:4d} / {total_all:4d} benar ({overall_pct:.2f}%)")
print("============================================================")

high_conf = data.get("errors_high_conf", [])
print(f"Total kesalahan deteksi dengan keyakinan tinggi (>= 85%): {len(high_conf)} file")

by_confusion = data.get("by_confusion", {})
if by_confusion:
    print("\nDetail Kesalahan Terbanyak (True -> Predicted):")
    for k, v in sorted(by_confusion.items(), key=lambda x: -len(x[1])):
        clean_key = k.replace("→", "->")
        print(f"  * {clean_key:<25} : {len(v):4d} file salah")
        # Print a few samples
        for i, item in enumerate(v[:3]):
            print(f"    - {item['file']} ({item['conf']*100:.1f}%)")
        if len(v) > 3:
            print(f"    ... dan {len(v)-3} file lainnya")
print("============================================================")
