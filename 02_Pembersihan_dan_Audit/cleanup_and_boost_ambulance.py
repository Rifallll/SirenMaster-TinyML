"""
cleanup_and_boost_ambulance.py
==============================
Step 1: Hapus semua file pendek (< 2 detik) dari seluruh kelas.
Step 2: Unduh sampel Ambulance berkualitas tinggi dari YouTube
        untuk mengganti 350 file yang dihapus dengan yang lebih gacor!
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, glob, subprocess, librosa, numpy as np, soundfile as sf

ROOT = r"C:\Users\ASUS\Videos\DATASET"
CLASSES = ["AMBULANCE", "FIRETRUCK", "NORMAL", "POLICE"]

# ======================================================
# STEP 1: HAPUS SEMUA FILE PENDEK (< 2 detik)
# ======================================================
print("=" * 65)
print("  STEP 1: MENGHAPUS FILE PENDEK (< 2 DETIK) DARI SEMUA KELAS")
print("=" * 65)

total_deleted = 0
for cls in CLASSES:
    folder = os.path.join(ROOT, cls)
    files = glob.glob(os.path.join(folder, "*.wav")) + glob.glob(os.path.join(folder, "*.mp3"))
    deleted = 0
    for f in files:
        try:
            y, sr = librosa.load(f, sr=None, duration=3.0)
            dur = librosa.get_duration(y=y, sr=sr)
            if dur < 2.0:
                os.remove(f)
                deleted += 1
        except:
            os.remove(f)  # Hapus juga file yang corrupt
            deleted += 1
    print(f"  [{cls}] Dihapus: {deleted} file pendek")
    total_deleted += deleted

print(f"\n  Total dihapus: {total_deleted} file pendek/rusak")

# ======================================================
# STEP 2: DOWNLOAD AMBULANCE BERKUALITAS TINGGI
# ======================================================
print()
print("=" * 65)
print("  STEP 2: DOWNLOAD AMBULANCE BERKUALITAS TINGGI SEBAGAI PENGGANTI")
print("=" * 65)

# Sumber ambulance paling bersih, jelas, dan panjang siklus gelombangnya
AMBULANCE_URLS = {
    "amb_uk_full_cycle":    "ytsearch1:UK ambulance siren sound effect full wail yelp 2 tone 10 minutes",
    "amb_usa_wail":         "ytsearch1:american ambulance siren wail sound effect long recording",
    "amb_euro_hi_lo":       "ytsearch1:european ambulance hi lo siren sound 10 minutes",
    "amb_uk_yelp":          "ytsearch1:british ambulance yelp siren nee naw sound effect long",
    "amb_germany_blue":     "ytsearch1:german ambulance siren sound effect long wail",
    "amb_france_twoton":    "ytsearch1:french ambulance two tone siren sound long recording",
    "amb_japan_siren":      "ytsearch1:japanese ambulance siren sound effect long",
    "amb_indonesia_baru":   "ytsearch1:suara ambulans indonesia sirine wail panjang",
}

AMB_DIR = os.path.join(ROOT, "AMBULANCE")

for name, url in AMBULANCE_URLS.items():
    print(f"\n[>] Mengunduh: {name} ...")
    out_template = os.path.join(AMB_DIR, f"dl_boost_{name}_%(id)s.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "0",
        "--download-sections", "*00:00:00-00:05:00",  # Ambil 5 menit
        "--no-playlist",
        "-o", out_template,
        url
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"  [!] Gagal: {e}")

# ======================================================
# STEP 3: POTONG FILE AMBULANCE BARU MENJADI 4 DETIK
# ======================================================
print()
print("=" * 65)
print("  STEP 3: MEMOTONG FILE AMBULANCE BARU MENJADI 4 DETIK")
print("=" * 65)

new_mp3s = glob.glob(os.path.join(AMB_DIR, "dl_boost_*.mp3"))
total_new = 0

for wf in new_mp3s:
    if "_part" in wf:
        continue
    print(f"  -> Memproses: {os.path.basename(wf)} ...")
    try:
        y, sr = librosa.load(wf, sr=8000)
        chunk_len = int(4.0 * sr)
        saved = 0
        for i in range(0, len(y) - chunk_len, chunk_len):
            chunk = y[i:i+chunk_len]
            rms = float(np.sqrt(np.mean(chunk**2)))
            if rms < 0.005:  # Skip jika terlalu hening (gap antar sirine)
                continue
            out_name = os.path.splitext(wf)[0] + f"_part{saved:04d}.wav"
            sf.write(out_name, chunk, sr)
            saved += 1
            total_new += 1
        print(f"     [OK] {saved} sampel baru.")
        os.remove(wf)
    except Exception as e:
        print(f"     [!] Error: {e}")

print()
print("=" * 65)
print(f"  [SUCCESS] Selesai!")
print(f"  File pendek dihapus  : {total_deleted}")
print(f"  Sampel baru ditambah : {total_new}")
print()
print("  Sekarang latih ulang model dengan:")
print("    python 04_Training_AI\\train_v2_fix_police.py")
print("=" * 65)
