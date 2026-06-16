"""
download_lokal_v2.py
====================
Mengunduh dataset sirine Indonesia SUPER LENGKAP dari YouTube.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
"""
Mencakup:
  - POLISI  : Sirine panjang + klakson tot-tot Patwal
  - PEMADAM : Sirine konstan + klakson pomp-pomp Damkar
  - AMBULANS: Sirine ninu-ninu + klakson tembak Ambulans
  - NORMAL  : Klakson kendaraan biasa + keramaian lalu lintas (agar tidak salah tebak)

Hanya menyimpan cuplikan 1 detik yang KERAS (RMS > threshold) agar
tidak ada suara hening atau obrolan yang bocor masuk ke dataset.
"""

import os
import subprocess
import librosa
import numpy as np
from scipy.io import wavfile

# ──────────────────────────────────────────────────────────────
# Konfigurasi
# ──────────────────────────────────────────────────────────────
OUT_DIR      = "DATASET_LOKAL"
SAMPLE_RATE  = 8000
CHUNK_LEN_S  = 1.024            # 8192 sampel – sama persis ESP32
CHUNK_SAMPS  = int(SAMPLE_RATE * CHUNK_LEN_S)
RMS_THR      = 0.06             # Buang yang terlalu pelan (hening/obrolan)
MAX_DUR_SEC  = 300              # Tolak video > 5 menit (vlog, berita panjang)

# ──────────────────────────────────────────────────────────────
# Master daftar unduhan
# Format: (folder_kelas, url_atau_kueri, label_singkat)
# ──────────────────────────────────────────────────────────────
DOWNLOADS = [

    # ─── POLISI ─────────────────────────────────────────────────
    ("POLICE", "https://www.youtube.com/watch?v=e-7RDekiYu4",        "pol_patwal_yg_diberikan"),
    ("POLICE", "ytsearch1:suara sirine patwal polisi indonesia jernih no copyright",  "pol_patwal_1"),
    ("POLICE", "ytsearch1:suara klakson tot tot patwal polisi escort",                "pol_tottot_1"),
    ("POLICE", "ytsearch1:suara sirine polisi whelp yelp indonesia",                  "pol_whelp_1"),
    ("POLICE", "ytsearch1:suara klakson polisi militer VIP indonesia",                "pol_vip_1"),
    ("POLICE", "ytsearch1:sirine polisi indonesia 1 jam looping",                     "pol_loop_1"),

    # ─── PEMADAM KEBAKARAN ───────────────────────────────────────
    ("FIRETRUCK", "ytsearch1:suara sirine pemadam kebakaran damkar indonesia keras",        "dam_sirine_1"),
    ("FIRETRUCK", "ytsearch1:suara klakson pomp pomp pemadam kebakaran buka jalan",        "dam_pompom_1"),
    ("FIRETRUCK", "ytsearch1:suara airhorn klakson telolet truk pemadam indonesia",        "dam_airhorn_1"),
    ("FIRETRUCK", "ytsearch1:suara sirine damkar DKI Jakarta lewat",                       "dam_dki_1"),
    ("FIRETRUCK", "ytsearch1:fire truck siren indonesia tanpa musik",                      "dam_en_1"),

    # ─── AMBULANS ─────────────────────────────────────────────────
    ("AMBULANCE", "ytsearch1:suara sirine ambulans ninu ninu indonesia jernih",            "amb_ninu_1"),
    ("AMBULANCE", "ytsearch1:suara klakson tembak ambulans gawat darurat minggir",        "amb_tembak_1"),
    ("AMBULANCE", "ytsearch1:suara sirine ambulans ritme cepat darurat indonesia",        "amb_cepat_1"),
    ("AMBULANCE", "ytsearch1:suara sirine ambulans jenazah pelan indonesia",              "amb_pelan_1"),
    ("AMBULANCE", "ytsearch1:ambulance siren indonesia 1 hour looping",                  "amb_loop_1"),

    # ─── NORMAL (suara yang BUKAN sirine, agar AI tidak salah tebak) ──
    ("NORMAL", "ytsearch1:suara klakson mobil truk bus di jalan raya indonesia",          "nor_klakson_1"),
    ("NORMAL", "ytsearch1:suara klakson telolet bus indonesia compilation",               "nor_telolet_1"),
    ("NORMAL", "ytsearch1:suara keramaian lalu lintas jakarta siang hari",               "nor_macet_1"),
    ("NORMAL", "ytsearch1:suara motor lewat kencang knalpot racing indonesia",            "nor_motor_1"),
    ("NORMAL", "ytsearch1:suara klakson truk pasir besar indonesia",                     "nor_truk_1"),
]

# ──────────────────────────────────────────────────────────────
# Utilitas
# ──────────────────────────────────────────────────────────────

def download_audio(query_or_url: str, out_mp3: str) -> bool:
    """Unduh audio ke file MP3 menggunakan yt-dlp."""
    cmd = [
        "python", "-m", "yt_dlp",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "0",
        "--match-filter", f"duration < {MAX_DUR_SEC}",
        "--no-playlist",
        "-o", out_mp3,
        query_or_url
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return os.path.exists(out_mp3)
    except Exception as e:
        print(f"    [ERROR] Download gagal: {e}")
        return False


def slice_and_save(mp3_path: str, cls_dir: str, prefix: str):
    """Potong MP3 jadi cuplikan 1 detik, buang yang hening."""
    try:
        y, _ = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)
        total = len(y) // CHUNK_SAMPS
        saved = 0
        for j in range(total):
            chunk = y[j * CHUNK_SAMPS : (j+1) * CHUNK_SAMPS]
            rms = float(np.sqrt(np.mean(chunk**2)))
            if rms < RMS_THR:
                continue                            # Buang – hening / obrolan
            out_path = os.path.join(cls_dir, f"{prefix}_part{j:04d}.wav")
            wavfile.write(out_path, SAMPLE_RATE, np.int16(chunk * 32767))
            saved += 1
        return saved
    except Exception as e:
        print(f"    [ERROR] Proses audio gagal: {e}")
        return 0


# ──────────────────────────────────────────────────────────────
# Eksekusi utama
# ──────────────────────────────────────────────────────────────
print("=" * 65)
print("  DATASET SIRINE INDONESIA V2 – SUPER LENGKAP")
print("=" * 65)

# Siapkan folder kelas
for cls in ["POLICE", "FIRETRUCK", "AMBULANCE", "NORMAL"]:
    os.makedirs(os.path.join(OUT_DIR, cls), exist_ok=True)

tally = {"POLICE": 0, "FIRETRUCK": 0, "AMBULANCE": 0, "NORMAL": 0}

for i, (cls, query, label) in enumerate(DOWNLOADS, 1):
    cls_dir = os.path.join(OUT_DIR, cls)
    tmp_mp3 = f"_tmp_{cls}_{label}.mp3"

    print(f"\n[{i:02d}/{len(DOWNLOADS)}] [{cls}] {label}")
    print(f"    Sumber : {query[:80]}")

    if not download_audio(query, tmp_mp3):
        print("    [SKIP] Gagal diunduh, lewati.")
        continue

    n = slice_and_save(tmp_mp3, cls_dir, f"v2_{label}")
    tally[cls] += n
    print(f"    [OK] {n} cuplikan sirine murni berhasil disimpan.")

    try:
        os.remove(tmp_mp3)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
# Laporan akhir
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SELESAI! Ringkasan Dataset Lokal V2:")
print("=" * 65)
grand = 0
for cls, n in tally.items():
    # Hitung total file di folder (termasuk unduhan sebelumnya)
    existing = len([f for f in os.listdir(os.path.join(OUT_DIR, cls)) if f.endswith('.wav')])
    print(f"  {cls:12s}: +{n:4d} baru  |  {existing:5d} total di folder")
    grand += existing
print(f"\n  TOTAL KESELURUHAN : {grand} cuplikan WAV")
print("=" * 65)
print("  Langkah berikutnya: jalankan train_lokal.py")
print("=" * 65)
