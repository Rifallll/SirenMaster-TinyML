import glob, os
from collections import defaultdict

print("=== ANALISIS REALITA DATASET SEKARANG ===\n")

# 1. Distribusi per kelas
cats = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
cat_counts = {}
for cat in cats:
    files = glob.glob(f'{cat}/*.wav')
    cat_counts[cat] = len(files)

max_count = max(cat_counts.values())
print("DISTRIBUSI KELAS (Imbalance):")
for cat, count in cat_counts.items():
    ratio = count / max_count
    bar = '|' * int(ratio * 30)
    print(f"  {cat:12s}: {count:5d} [{bar:<30s}] {ratio:.1%}")

print()

# 2. Analisis POLICE - asal rekaman
police_files = glob.glob('POLICE/*.wav')
groups = defaultdict(int)
for f in police_files:
    base = os.path.basename(f)
    if 'police_0002' in base:
        groups['police_0002 (1 rekaman saja!)'] += 1
    elif 'police_0413' in base:
        groups['police_0413'] += 1
    elif base.startswith('aug_'):
        groups['augmented (lain)'] += 1
    elif base.startswith('yt_'):
        groups['youtube'] += 1
    elif base.startswith('tambah_'):
        groups['tambah_dataset'] += 1
    else:
        groups['other'] += 1

print("POLICE - Distribusi asal rekaman:")
for g, count in sorted(groups.items()):
    print(f"  {g}: {count} file")

print()

# 3. Analisis AMBULANCE
amb_files = glob.glob('AMBULANCE/*.wav')
amb_groups = defaultdict(int)
for f in amb_files:
    base = os.path.basename(f)
    if base.startswith('aug_noise'):
        amb_groups['aug_noise'] += 1
    elif base.startswith('aug_pitch'):
        amb_groups['aug_pitch'] += 1
    elif base.startswith('aug_stretch'):
        amb_groups['aug_stretch'] += 1
    elif base.startswith('yt_'):
        amb_groups['youtube'] += 1
    elif base.startswith('sound_'):
        amb_groups['sound (freesound.org)'] += 1
    elif base.startswith('ambulance_'):
        amb_groups['raw ambulance'] += 1
    else:
        amb_groups['other'] += 1

print("AMBULANCE - Distribusi asal rekaman:")
for g, count in sorted(amb_groups.items()):
    print(f"  {g}: {count} file")

print()

# 4. Masalah utama: POLICE terlalu didominasi 1 rekaman (police_0002)
police_0002_count = groups.get('police_0002 (1 rekaman saja!)', 0)
total_police = cat_counts['POLICE']
print("=== DIAGNOSA MASALAH UTAMA ===")
print(f"POLICE total: {total_police} file")
pct_0002 = police_0002_count / total_police * 100 if total_police > 0 else 0
print(f"Dari sumber police_0002 SAJA: {police_0002_count} file ({pct_0002:.1f}%)")
print()
if pct_0002 > 40:
    print("[MASALAH KRITIS] Lebih dari 40% data POLICE berasal dari 1 rekaman saja!")
    print("  -> Model belajar SUARA KHUSUS itu, bukan karakter sirine polisi secara umum.")
    print("  -> Di dunia nyata, sirine polisi berbeda merek/tipe akan gagal dideteksi!")
print()
print(f"POLICE ({total_police}) vs AMBULANCE ({cat_counts['AMBULANCE']}) vs NORMAL ({cat_counts['NORMAL']})")
print(f"NORMAL hampir {cat_counts['NORMAL']//total_police}x lebih banyak dari POLICE -> bias ke NORMAL")
