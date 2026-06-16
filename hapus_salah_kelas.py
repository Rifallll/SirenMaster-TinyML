"""
HAPUS FILE YANG JELAS SALAH KLASIFIKASI
Berdasarkan audit: hapus per-grup yang terbukti tidak sesuai kelasnya
"""
import os, sys, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = r"C:\Users\ASUS\Videos\DATASET"

# =====================================================================
# DAFTAR GRUP YANG JELAS SALAH — BERDASARKAN HASIL AUDIT
# Format: (folder, [prefix/keyword file yang akan dihapus])
# =====================================================================
HAPUS = {

    # ── FIRETRUCK → terdeteksi AMBULANCE (bukan suara firetruck) ──
    # Grup fire_0047, fire_0189, fire_0191, fire_0210, fire_0222 semua
    # terdeteksi 99.6% sebagai AMBULANCE → kemungkinan salah label
    "FIRETRUCK": [
        "fire_0047_",
        "fire_0189_",
        "fire_0191_",
        "fire_0210_",
        "fire_0222_",
        "fire_0166_",
        # FIRETRUCK → terdeteksi NORMAL (bukan sirine, cuma suara mesin/klakson)
        "dam_extra_1_",
        "dam_extra_2_",
        "dam_extra_4_",
        "dam_extra_5_",
        "dam_extra_7_",
        "yt_indo_1_part_",
        # Rekaman guru yang salah
        "guru_firetruck_1780558406696",
        "guru_firetruck_1780558653705",
    ],

    # ── POLICE → terdeteksi AMBULANCE (suaranya mirip ambulance) ──
    "POLICE": [
        # Rekaman guru yang salah deteksi
        "guru_police_otomatis_1781057448",
        "guru_police_otomatis_1781057190",
        "guru_police_otomatis_1781057611",
        # Sirine polisi versi patwal/tottot yang terlalu pelan → NORMAL
        "v2_pol_patwal_1_part",
        "v2_pol_tottot_1_part",
        # YouTube indo yang isinya bukan sirine polisi jelas
        "yt_indo_0_part_",
    ],

    # ── NORMAL → terdeteksi AMBULANCE (ada suara sirine di dalamnya) ──
    "NORMAL": [
        "noise_u4_69598-4-2-0",
        "noise_u9_115241-9-0-8",
        "noise_u9_194733-9-0-14",
        "noise_u7_105029-7-2-14",
        "urban_9_194733-9-0-11",
        "urban_7_105029-7-2-14",
        "urban_2_27070-2-0-3",
        "urban_2_97331-2-0-55",
        "urban_2_116423-2-0-4",
    ],

    # ── AMBULANCE → terdeteksi NORMAL (sirine terlalu pelan/jauh) ──
    "AMBULANCE": [
        "v2_amb_pelan_1_part",
        "tambah_1780392496",
    ],
}

# Hitung dulu berapa yang akan dihapus
print("=" * 60)
print("  HAPUS FILE SALAH KLASIFIKASI — BERDASARKAN AUDIT")
print("=" * 60)

total_plan = 0
plan = {}  # folder → list filepath

for folder, prefixes in HAPUS.items():
    folder_path = os.path.join(BASE, folder)
    files = os.listdir(folder_path)
    to_del = []
    for fname in files:
        for prefix in prefixes:
            if fname.startswith(prefix) or prefix in fname:
                to_del.append(os.path.join(folder_path, fname))
                break
    plan[folder] = to_del
    total_plan += len(to_del)
    print(f"\n  [{folder}] → {len(to_del)} file akan dihapus:")
    for fp in to_del[:5]:
        print(f"    - {os.path.basename(fp)}")
    if len(to_del) > 5:
        print(f"    ... dan {len(to_del)-5} file lainnya")

print(f"\n{'='*60}")
print(f"  TOTAL: {total_plan} file")
print(f"{'='*60}")
print(f"\nLanjutkan hapus? (YA/tidak): ", end="")

if input().strip().upper() != "YA":
    print("[BATAL]")
    sys.exit(0)

print("\n[*] Menghapus...")
ok = 0
err = 0
for folder, filepaths in plan.items():
    for fp in filepaths:
        try:
            os.remove(fp)
            ok += 1
        except Exception as e:
            print(f"  [ERR] {os.path.basename(fp)}: {e}")
            err += 1

print(f"\n{'='*60}")
print(f"  SELESAI — {ok} file dihapus, {err} gagal")
print(f"{'='*60}")

# Sisa dataset
print("\n  Sisa dataset:")
for cat in ['AMBULANCE','FIRETRUCK','NORMAL','POLICE']:
    d = os.path.join(BASE, cat)
    n = len([f for f in os.listdir(d) if f.lower().endswith(('.wav','.m4a'))])
    print(f"  {cat:12s}: {n} file")

print("\n✅ Jalankan: python train_lokal.py")
