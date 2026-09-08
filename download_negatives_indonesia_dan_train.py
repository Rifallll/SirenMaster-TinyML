"""
download_negatives_indonesia_dan_train.py
=========================================
Download dataset ESC-50 (Environmental Sound Classification) dari GitHub
dan ambil suara-suara yang sering bocor sebagai sirine:
- Crowd noise / keramaian
- Baby crying / tangisan bayi
- Laughing / tawa
- Clapping / tepuk tangan
- Church bells (mirip sirine)
- Engine / mesin
- Various speech-like sounds
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, time, zipfile, shutil
import numpy as np
import librosa
import scipy.io.wavfile as wav

try:
    import requests
except ImportError:
    os.system("pip install requests")
    import requests

SAMPLE_RATE  = 8000
DURATION     = 4.0
TARGET_LEN   = int(SAMPLE_RATE * DURATION)
NORMAL_DIR   = r"C:\Users\ASUS\Videos\DATASET\NORMAL"
DOWNLOAD_DIR = r"C:\Users\ASUS\Videos\DATASET\TRASH\esc50_download"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(NORMAL_DIR, exist_ok=True)

# ── Kategori ESC-50 yang relevan sebagai "Hard Negative" sirine ──
# Lihat: https://github.com/karolpiczak/ESC-50#dataset-overview
RELEVANT_CATEGORIES = {
    "crying_baby":    "bayi_nangis",       # Sering bocor sebagai sirine!
    "laughing":       "tawa",
    "clapping":       "tepuk_tangan",
    "crowd":          "keramaian",         # Paling relevan!
    "church_bells":   "lonceng_gereja",    # Mirip sirine
    "engine":         "suara_mesin",
    "car_horn":       "klakson",           # Sudah ada tapi perkuat
    "sneezing":       "bersin",
    "drinking_sipping": "suara_minum",
    "pouring_water":  "air_mengalir",
    "keyboard_typing": "ketik_keyboard",
    "footsteps":      "langkah_kaki",
    "door_wood_knock": "ketukan_pintu",
}

print("=" * 65)
print("  DOWNLOADER ESC-50 DATASET (HARD NEGATIVES OTOMATIS)")
print("=" * 65)
print()

# ── Step 1: Download ESC-50 dari GitHub ──
zip_path = os.path.join(DOWNLOAD_DIR, "ESC-50-master.zip")
extract_path = os.path.join(DOWNLOAD_DIR, "ESC-50-master")

if not os.path.exists(zip_path):
    print("[1/3] Mengunduh ESC-50 dari GitHub (~40MB)...")
    url = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"    Mencoba sambungan (Percobaan {attempt+1}/{max_retries})...")
            # Tambahkan timeout lebih besar
            r = requests.get(url, stream=True, timeout=600)
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            downloaded_bytes = 0
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        if total > 0:
                            pct = downloaded_bytes / total * 100
                            print(f"    Progress: {pct:.0f}% ({downloaded_bytes//1024//1024}MB/{total//1024//1024}MB)    ", end='\r')
            print(f"\n    ✅ Download selesai: {downloaded_bytes//1024//1024} MB")
            break # Berhasil, keluar dari loop
        except Exception as e:
            print(f"\n    ⚠️ Gagal pada percobaan {attempt+1}: {e}")
            if attempt < max_retries - 1:
                print("    Menunggu 5 detik sebelum mencoba lagi...")
                time.sleep(5)
            else:
                print("    ❌ Gagal total setelah 3 kali percobaan. Cek koneksi internet Anda.")
                sys.exit(1)
else:
    print("[1/3] File ESC-50 sudah ada, skip download.")

# ── Step 2: Extract ZIP ──
print("\n[2/3] Mengekstrak file ZIP...")
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DOWNLOAD_DIR)
    print(f"    ✅ Diekstrak ke: {extract_path}")
else:
    print("    File sudah diekstrak, skip.")

# ── Step 3: Proses dan simpan ke NORMAL ──
print("\n[3/3] Memproses dan menyimpan segmen ke folder NORMAL...")
audio_dir = os.path.join(extract_path, "audio")
meta_path = os.path.join(extract_path, "meta", "esc50.csv")

if not os.path.exists(audio_dir):
    print(f"❌ Folder audio tidak ditemukan: {audio_dir}")
    sys.exit(1)

# Baca metadata
categories_map = {}
try:
    with open(meta_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                filename = parts[0]
                category = parts[3]
                categories_map[filename] = category
except Exception as e:
    print(f"⚠️ Tidak bisa baca metadata: {e}. Lanjut tanpa filter kategori.")

ts = int(time.time())
total_saved = 0
cat_counts = {}

audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
print(f"    Total file di ESC-50: {len(audio_files)}")

for fname in sorted(audio_files):
    fpath = os.path.join(audio_dir, fname)
    category = categories_map.get(fname, "unknown")

    # Cek apakah kategori ini relevan
    label = RELEVANT_CATEGORIES.get(category, None)
    if label is None:
        continue  # Skip kategori yang tidak relevan

    try:
        y, sr = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)
        # ESC-50 semua 5 detik, kita ambil 4 detik awal
        if len(y) < TARGET_LEN:
            continue
        chunk = y[:TARGET_LEN]

        # Auto-gain normalisasi
        max_val = np.max(np.abs(chunk))
        if max_val > 0.001:
            chunk = chunk * (0.80 / max_val)

        # Nama file dengan prefix hard_neg agar masuk VIP training
        out_name = f"hard_neg_{label}_{ts}_{total_saved:04d}.wav"
        out_path = os.path.join(NORMAL_DIR, out_name)
        wav.write(out_path, SAMPLE_RATE, (chunk * 32767).astype(np.int16))

        total_saved += 1
        cat_counts[label] = cat_counts.get(label, 0) + 1

    except Exception as e:
        pass  # Skip file bermasalah

print(f"\n    ✅ Total tersimpan: {total_saved} file ke NORMAL/")
print("\n    Rincian per kategori:")
for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"      - {cat}: {count} file")

if total_saved == 0:
    print("\n❌ Tidak ada file tersimpan. Cek path ESC-50.")
    sys.exit(1)

print(f"\n{'='*65}")
print(f"  ✅ SELESAI! {total_saved} suara hard negative baru ditambahkan!")
print(f"  Siap untuk training ulang AI.")
print(f"{'='*65}\n")

# ── Auto Training ──
print("=" * 65)
print("  MEMULAI TRAINING ULANG AI...")
print("=" * 65)
import subprocess
train_script = r"C:\Users\ASUS\Videos\DATASET\04_Training_AI\train_lokal.py"
result = subprocess.run([sys.executable, train_script])
if result.returncode == 0:
    print("\n✅ TRAINING SELESAI! Silakan Upload ke Arduino IDE!")
else:
    print("\n⚠️ Training gagal. Jalankan manual: python 04_Training_AI\\train_lokal.py")
