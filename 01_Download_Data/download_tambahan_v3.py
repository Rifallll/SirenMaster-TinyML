"""
download_tambahan_v3.py
=======================
Download data sirine LANGSUNG ke folder dataset utama.
Fokus pada kelas yang paling lemah: POLICE (butuh variasi lebih banyak).
Juga tambahkan AMBULANCE dan FIRETRUCK dari sumber baru.

Target:
- POLICE    : +300-500 file baru (variasi sirine patwal indonesia)
- AMBULANCE : +200 file baru (variasi sirine ninu-ninu)
- FIRETRUCK : +150 file baru
- NORMAL    : +100 file baru (untuk balancing)

Cara jalankan:
    uv run python 01_Download_Data/download_tambahan_v3.py
"""

import os, sys, subprocess, hashlib, glob
import numpy as np
import librosa
from scipy.io import wavfile

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── KONFIGURASI ──────────────────────────────────────────────────────
BASE_DIR     = r"C:\Users\ASUS\Videos\DATASET"
SAMPLE_RATE  = 8000
CHUNK_LEN_S  = 4.0              # 4 detik per cuplikan (sama dengan model)
CHUNK_SAMPS  = int(SAMPLE_RATE * CHUNK_LEN_S)
RMS_THR      = 0.04             # Buang yang hening/obrolan (turunkan dari 0.06 agar tidak terlalu ketat)
MAX_DUR_SEC  = 600              # Tolak video > 10 menit
HASH_FILE    = os.path.join(BASE_DIR, "downloaded_hashes.txt")
FFMPEG_PATH  = r"C:\Users\ASUS\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin"

# ── SUMBER DOWNLOAD BARU ─────────────────────────────────────────────
# Format: (folder_kelas, query_atau_url, label_singkat)
# FOKUS: banyak variasi sirine polisi Indonesia (ini yang paling kurang)
DOWNLOADS = [

    # ─── POLICE: Variasi Baru ────────────────────────────────────────
    ("POLICE", "ytsearch1:suara sirine patwal polisi indonesia original", "pol_new_01"),
    ("POLICE", "ytsearch1:sirine mobil polisi indonesia lewat jalan", "pol_new_02"),
    ("POLICE", "ytsearch1:suara sirine polisi yelp wail indonesia", "pol_new_03"),
    ("POLICE", "ytsearch1:sirine polisi indonesia lengkap semua jenis", "pol_new_04"),
    ("POLICE", "ytsearch1:suara polisi indonesia klakson sirine escort VIP", "pol_new_05"),
    ("POLICE", "ytsearch1:suara sirine patwal polisi brigade mobil brimob", "pol_new_06"),
    ("POLICE", "ytsearch1:sirine polisi indonesia 2024 jernih original", "pol_new_07"),
    ("POLICE", "ytsearch1:suara klakson toa polisi lalu lintas indonesia", "pol_new_08"),
    ("POLICE", "ytsearch1:konvoi polisi sirine indonesia", "pol_new_09"),
    ("POLICE", "ytsearch1:sound effect sirine polisi indonesia hd", "pol_new_10"),
    ("POLICE", "ytsearch1:suara sirine polisi motor patwal", "pol_new_11"),
    ("POLICE", "ytsearch1:escort presiden sirine polisi indonesia", "pol_new_12"),

    # ─── AMBULANCE: Variasi Baru ─────────────────────────────────────
    ("AMBULANCE", "ytsearch1:suara sirine ambulans puskesmas indonesia", "amb_new_01"),
    ("AMBULANCE", "ytsearch1:suara ambulans gawat darurat indonesia baru", "amb_new_02"),
    ("AMBULANCE", "ytsearch1:suara sirine ambulans rumah sakit lewat jalan", "amb_new_03"),
    ("AMBULANCE", "ytsearch1:suara ninu ninu ambulans asli indonesia", "amb_new_04"),
    ("AMBULANCE", "ytsearch1:suara ambulans RI 118 panggil gawat darurat", "amb_new_05"),
    ("AMBULANCE", "ytsearch1:sirine ambulans jenazah yayasan indonesia", "amb_new_06"),

    # ─── FIRETRUCK: Variasi Baru ─────────────────────────────────────
    ("FIRETRUCK", "ytsearch1:suara sirine pemadam kebakaran jakarta terbaru", "dam_new_01"),
    ("FIRETRUCK", "ytsearch1:suara damkar indonesia lewat jalan sirine kencang", "dam_new_02"),
    ("FIRETRUCK", "ytsearch1:suara klakson pomp damkar indonesia asli", "dam_new_03"),
    ("FIRETRUCK", "ytsearch1:fire engine siren indonesia original sound", "dam_new_04"),

    # ─── NORMAL: Suara Bukan Sirine (Penting untuk hard negative) ────
    ("NORMAL", "ytsearch1:suara kemacetan jakarta siang hari tanpa sirine", "nor_new_01"),
    ("NORMAL", "ytsearch1:suara knalpot motor kencang indonesia street racing", "nor_new_02"),
    ("NORMAL", "ytsearch1:suara hujan lebat dan angin kencang indonesia", "nor_new_03"),
    ("NORMAL", "ytsearch1:ambient sound pasar pagi indonesia ramai", "nor_new_04"),
]

# ── LOAD HASH YANG SUDAH DIDOWNLOAD ─────────────────────────────────
downloaded_hashes = set()
if os.path.exists(HASH_FILE):
    with open(HASH_FILE, 'r') as f:
        downloaded_hashes = set(line.strip() for line in f if line.strip())
    print(f"[INFO] {len(downloaded_hashes)} URL sebelumnya sudah didownload, akan di-skip.")

def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def save_hash(url: str):
    h = get_url_hash(url)
    downloaded_hashes.add(h)
    with open(HASH_FILE, 'a') as f:
        f.write(h + "\n")

def count_existing(cat: str) -> int:
    return len(glob.glob(os.path.join(BASE_DIR, cat, "*.wav")))

def download_audio(query_or_url: str, out_mp3: str) -> bool:
    cmd = [
        "python", "-m", "yt_dlp",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "0",
        "--match-filter", f"duration < {MAX_DUR_SEC}",
        "--no-playlist",
        "--quiet", "--no-warnings",
        "--ffmpeg-location", FFMPEG_PATH,
        "-o", out_mp3,
        query_or_url
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return os.path.exists(out_mp3)
    except Exception as e:
        print(f"    [ERROR] Download gagal: {e}")
        return False

def slice_and_save(mp3_path: str, cls_dir: str, prefix: str) -> int:
    try:
        y, _ = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)
        total_chunks = len(y) // CHUNK_SAMPS
        saved = 0
        for j in range(total_chunks):
            chunk = y[j * CHUNK_SAMPS : (j+1) * CHUNK_SAMPS]
            rms = float(np.sqrt(np.mean(chunk**2)))
            if rms < RMS_THR:
                continue
            # Simpan langsung ke folder utama dataset
            out_path = os.path.join(cls_dir, f"dl_v3_{prefix}_part{j:04d}.wav")
            if not os.path.exists(out_path):
                wavfile.write(out_path, SAMPLE_RATE, np.int16(chunk * 32767))
                saved += 1
        return saved
    except Exception as e:
        print(f"    [ERROR] Slice audio gagal: {e}")
        return 0

# ── MAIN ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  DOWNLOAD SIRINE INDONESIA v3 — LANGSUNG KE DATASET UTAMA")
print("=" * 65)
print()
print("Jumlah file saat ini:")
for cat in ["POLICE", "FIRETRUCK", "AMBULANCE", "NORMAL"]:
    print(f"  {cat:12s}: {count_existing(cat)} file")
print()

tally = {cat: 0 for cat in ["POLICE", "FIRETRUCK", "AMBULANCE", "NORMAL"]}

for i, (cls, query, label) in enumerate(DOWNLOADS, 1):
    url_hash = get_url_hash(query)
    if url_hash in downloaded_hashes:
        print(f"[{i:02d}/{len(DOWNLOADS)}] [SKIP] {label} sudah didownload sebelumnya.")
        continue

    cls_dir = os.path.join(BASE_DIR, cls)
    tmp_mp3 = os.path.join(BASE_DIR, f"_tmp_{label}.mp3")

    print(f"[{i:02d}/{len(DOWNLOADS)}] [{cls}] Downloading: {label}...")
    print(f"    Query: {query[:70]}")

    if not download_audio(query, tmp_mp3):
        print(f"    [GAGAL] Tidak bisa download. Skip.")
        continue

    n = slice_and_save(tmp_mp3, cls_dir, label)
    tally[cls] += n
    print(f"    [OK] +{n} cuplikan 4 detik disimpan ke {cls}/")
    save_hash(query)

    try:
        os.remove(tmp_mp3)
    except Exception:
        pass

print()
print("=" * 65)
print("  SELESAI! Ringkasan Penambahan Data:")
print("=" * 65)
for cat in ["POLICE", "FIRETRUCK", "AMBULANCE", "NORMAL"]:
    after = count_existing(cat)
    added = tally[cat]
    print(f"  {cat:12s}: +{added:4d} baru  |  {after:5d} total")
print()
print("LANGKAH BERIKUTNYA:")
print("  1. Hapus cache lama (sudah otomatis kalau ada)")
print("  2. Jalankan: uv run python 04_Training_AI/train_v2_fix_police.py")
print("=" * 65)

# Hapus cache jika ada penambahan data
if any(v > 0 for v in tally.values()):
    import glob as gl
    for cache in gl.glob(os.path.join(BASE_DIR, "*cache*.npz")):
        try:
            os.remove(cache)
            print(f"[INFO] Cache '{os.path.basename(cache)}' dihapus.")
        except:
            pass
