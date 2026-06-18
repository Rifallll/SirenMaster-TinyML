"""
AUDIT CERDAS — Kelompokkan file bermasalah berdasarkan SUMBER REKAMAN
Tampilkan grup mana yang KONSISTEN salah (bukan cuma 1 file)
Jangan hapus dulu — cukup laporan untuk review manual
"""
import json, os, sys, re
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATASET_BASE  = r"C:\Users\ASUS\Videos\DATASET"
EVAL_PATH     = os.path.join(DATASET_BASE, "eval_terbaru.json")
CONF_MIN      = 0.90

with open(EVAL_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

errors = [e for e in data.get("errors_high_conf", []) if e['conf'] >= CONF_MIN]

def get_source_group(filename):
    """Ambil nama sumber rekaman (tanpa _loud, _original, _noise_light, _shift, aug_, dll)"""
    name = os.path.splitext(filename)[0]
    # Hapus prefix augmentasi
    name = re.sub(r'^aug_[a-zA-Z]+_(\d+_)?', '', name)
    # Hapus suffix varian
    name = re.sub(r'_(loud|original|noise_light|shift|seg\d+|part\d+)$', '', name)
    name = re.sub(r'_\d+$', '', name)
    return name

# Kelompokkan per (folder, source_group, prediksi)
groups = defaultdict(list)
for e in errors:
    src = get_source_group(e['file'])
    key = (e['folder'], src, e['pred'])
    groups[key].append(e)

# Urutkan: terbanyak file dulu
sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))

print("=" * 70)
print("  AUDIT DATASET — GRUP REKAMAN YANG KONSISTEN SALAH")
print(f"  (Hanya file dengan keyakinan salah >= {CONF_MIN*100:.0f}%)")
print("=" * 70)

# Pisah berdasarkan arah kesalahan
directions = defaultdict(list)
for (folder, src, pred), items in sorted_groups:
    if len(items) >= 2:  # Hanya tampilkan yang >= 2 file salah
        directions[f"{folder} → {pred}"].append((src, items))

for direction, groups_list in sorted(directions.items()):
    total_files = sum(len(items) for _, items in groups_list)
    print(f"\n{'='*70}")
    print(f"  ❌ [{direction}] — {len(groups_list)} grup, {total_files} file total")
    print(f"{'='*70}")
    
    for src, items in sorted(groups_list, key=lambda x: -len(x[1]))[:20]:
        avg_conf = sum(i['conf'] for i in items) / len(items)
        sample_files = [i['file'] for i in items[:3]]
        print(f"\n  Grup: '{src}' ({len(items)} file, rata-rata yakin: {avg_conf*100:.0f}%)")
        for f in sample_files:
            print(f"    - {f}")
        if len(items) > 3:
            print(f"    ... dan {len(items)-3} file lainnya")

print(f"\n\n{'='*70}")
print(f"  RINGKASAN GRUP BERMASALAH (>= 2 file salah yakin >= 90%):")
print(f"{'='*70}")
for direction, groups_list in sorted(directions.items()):
    total = sum(len(items) for _, items in groups_list)
    print(f"  {direction:35s}: {len(groups_list):4d} grup, {total:5d} file")
