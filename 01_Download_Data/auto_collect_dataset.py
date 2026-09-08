"""
auto_collect_dataset.py
=======================
Otomatis download audio sirine dari YouTube,
klasifikasi dengan model AI, simpan ke folder yang benar.
- Skip duplikat (hash MD5)
- Hanya simpan jika confidence > SAVE_THRESHOLD
- Tidak akan menimpa atau salah folder
"""

import os
import sys
import hashlib
import tempfile
import numpy as np
import tensorflow as tf
import librosa
import scipy.io.wavfile as wav
import yt_dlp
import subprocess
import json
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ─── KONFIGURASI ────────────────────────────────────────────────
DATASET_ROOT   = r"C:\Users\ASUS\Videos\DATASET"
MODEL_PATH     = os.path.join(DATASET_ROOT, "siren_model_quant.tflite")
CACHE_PATH     = os.path.join(DATASET_ROOT, "siren_40x63_melspec_cache_lokal.npz")
HASH_DB_PATH   = os.path.join(DATASET_ROOT, "downloaded_hashes.txt")

CATEGORIES     = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
SAMPLE_RATE    = 8000
DURATION       = 1.024
CHUNK_SAMPLES  = int(SAMPLE_RATE * DURATION)

# Hanya simpan jika AI yakin di atas ambang ini
SAVE_THRESHOLD     = 0.90  # 90% keyakinan minimum untuk disimpan
MAX_VIDEO_DURATION = 600   # Maks 10 menit (detik)
MAX_FILESIZE_MB    = 60    # Maks 60 MB per file

# Posterior adjustment (sama dengan test_laptop.py)
ADJUSTMENT = np.array([0.82, 1.0, 1.05, 1.25])

# Target jumlah file baru per kategori (berhenti jika sudah cukup)
TARGET_NEW_PER_CAT = 300

# ─── DAFTAR PENCARIAN YOUTUBE ────────────────────────────────────
# Format: (query_pencarian, kategori_target)
SEARCH_QUERIES = [
    # ── AMBULANCE ──
    ("suara sirine ambulans Indonesia", "AMBULANCE"),
    ("ambulance siren sound Indonesia", "AMBULANCE"),
    ("sirine ambulan kencang", "AMBULANCE"),
    ("ambulance Indonesia horn sound", "AMBULANCE"),
    ("suara ambulans 118 Indonesia", "AMBULANCE"),
    ("ambulance siren wail yelp", "AMBULANCE"),
    ("suara sirene ziekenwagen", "AMBULANCE"),
    ("ambulance sound effect", "AMBULANCE"),
    
    # ── POLICE ──
    ("suara sirine polisi Indonesia", "POLICE"),
    ("police siren sound Indonesia", "POLICE"),
    ("sirine polri Indonesia", "POLICE"),
    ("suara sirine mobil polisi", "POLICE"),
    ("police car siren hi-lo", "POLICE"),
    ("suara sirine kepolisian", "POLICE"),
    ("wail yelp police siren sound effect", "POLICE"),
    ("suara sirene politie", "POLICE"),
    
    # ── FIRETRUCK ──
    ("suara sirine pemadam kebakaran Indonesia", "FIRETRUCK"),
    ("fire truck siren Indonesia", "FIRETRUCK"),
    ("sirine damkar Indonesia", "FIRETRUCK"),
    ("suara mobil pemadam kebakaran", "FIRETRUCK"),
    ("firetruck siren sound effect", "FIRETRUCK"),
    ("fire engine siren", "FIRETRUCK"),
    
    # ── NORMAL (suara jalanan) ──
    ("suara lalu lintas jalan raya Indonesia", "NORMAL"),
    ("traffic noise Indonesia city", "NORMAL"),
    ("ambient road sound Jakarta", "NORMAL"),
    ("suara jalanan kota Indonesia", "NORMAL"),
]

# ─── LOAD MODEL & NORMALISASI ────────────────────────────────────
print("=" * 60)
print("  AUTO DATASET COLLECTOR - AI Siren")
print("=" * 60)
print("\n[*] Memuat model AI...")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

with np.load(CACHE_PATH, allow_pickle=True) as data:
    X_train = data['X_train']
    global_mean = np.mean(X_train, axis=(0, 1))
    global_std  = np.std(X_train,  axis=(0, 1))
    global_std[global_std < 1e-6] = 1.0
print("[OK] Model dan normalisasi dimuat.")

# ─── LOAD HASH DATABASE (ANTI DUPLIKAT) ─────────────────────────
existing_hashes = set()
if os.path.exists(HASH_DB_PATH):
    with open(HASH_DB_PATH, 'r') as f:
        existing_hashes = set(line.strip() for line in f if line.strip())
print(f"[*] Hash duplikat yang sudah ada: {len(existing_hashes)}")

def save_hash(h):
    existing_hashes.add(h)
    with open(HASH_DB_PATH, 'a') as f:
        f.write(h + '\n')

def file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def audio_hash(y):
    """Hash dari sinyal audio (untuk cek duplikat konten)"""
    y_rounded = np.round(y * 1000).astype(np.int16)
    return hashlib.md5(y_rounded.tobytes()).hexdigest()

# ─── FUNGSI EKSTRAK & INFERENSI ──────────────────────────────────
def extract_melspec(y, sr=8000):
    if len(y) < CHUNK_SAMPLES:
        y = np.pad(y, (0, CHUNK_SAMPLES - len(y)))
    else:
        y = y[:CHUNK_SAMPLES]
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=256, hop_length=128,
        n_mels=40, fmin=0, fmax=4000,
        window='hamming', center=False
    )
    return np.log(S + 1e-9).astype(np.float32).T  # (63, 40)

def classify_chunk(y):
    """Klasifikasi 1 chunk audio. Return (category_idx, confidence)"""
    feat = extract_melspec(y)
    feat_scaled = (feat - global_mean) / global_std
    feat_scaled = feat_scaled[np.newaxis, :, :, np.newaxis].astype(np.float32)
    
    sc, zp = inp_det['quantization']
    if inp_det['dtype'] == np.int8:
        feat_q = np.clip(np.round(feat_scaled / sc) + zp, -128, 127).astype(np.int8)
    else:
        feat_q = feat_scaled
    
    interpreter.set_tensor(inp_det['index'], feat_q)
    interpreter.invoke()
    raw = interpreter.get_tensor(out_det['index'])
    
    osc, ozp = out_det['quantization']
    if out_det['dtype'] == np.int8:
        probs = (raw[0].astype(np.float32) - ozp) * osc
    else:
        probs = raw[0]
    
    probs_adj = probs * ADJUSTMENT
    probs_adj = probs_adj / probs_adj.sum()
    idx = int(np.argmax(probs_adj))
    return idx, float(probs_adj[idx])

# ─── HITUNG FILE YANG SUDAH ADA ──────────────────────────────────
def count_existing():
    counts = {}
    for cat in CATEGORIES:
        total = 0
        for base in [DATASET_ROOT,
                     os.path.join(DATASET_ROOT, "DATASET_LOKAL"),
                     os.path.join(DATASET_ROOT, "TAMBAH_DATASET")]:
            d = os.path.join(base, cat)
            if os.path.exists(d):
                total += sum(1 for f in os.listdir(d) if f.endswith('.wav'))
        counts[cat] = total
    return counts

existing_counts = count_existing()
new_counts = {cat: 0 for cat in CATEGORIES}

print("\n[*] File existing per kategori:")
for cat in CATEGORIES:
    print(f"    {cat:<12}: {existing_counts[cat]:>5} file")

# ─── DIREKTORI SIMPAN ────────────────────────────────────────────
SAVE_BASE = os.path.join(DATASET_ROOT, "TAMBAH_DATASET")
for cat in CATEGORIES:
    os.makedirs(os.path.join(SAVE_BASE, cat), exist_ok=True)

# ─── FUNGSI DOWNLOAD & PROSES ────────────────────────────────────
def process_and_save(audio_path, expected_cat, source_label):
    """Load audio, potong jadi chunk, klasifikasi, simpan jika cocok."""
    saved = 0
    skipped_dup = 0
    skipped_wrong = 0
    skipped_lowconf = 0
    
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"    [!] Gagal load audio: {e}")
        return 0, 0, 0, 0
    
    # Potong jadi chunk non-overlapping
    hop = CHUNK_SAMPLES
    for start in range(0, len(y) - CHUNK_SAMPLES + 1, hop):
        chunk = y[start:start + CHUNK_SAMPLES]
        
        # Skip hening
        rms = np.sqrt(np.mean(chunk**2))
        if rms < 0.002:
            continue
        
        # Cek duplikat konten
        h = audio_hash(chunk)
        if h in existing_hashes:
            skipped_dup += 1
            continue
        
        # Klasifikasi
        cat_idx, conf = classify_chunk(chunk)
        cat_name = CATEGORIES[cat_idx]
        
        # Verifikasi: harus sesuai expected_cat ATAU AI sangat yakin
        if cat_name != expected_cat:
            skipped_wrong += 1
            continue
        
        if conf < SAVE_THRESHOLD:
            skipped_lowconf += 1
            continue
        
        # Simpan!
        ts = int(time.time() * 1000) % 1000000
        fname = f"web_{cat_name.lower()}_{ts}_{saved:03d}.wav"
        fpath = os.path.join(SAVE_BASE, cat_name, fname)
        
        chunk_int16 = np.int16(chunk / max(np.max(np.abs(chunk)), 1e-6) * 32767)
        wav.write(fpath, SAMPLE_RATE, chunk_int16)
        save_hash(h)
        new_counts[cat_name] += 1
        saved += 1
    
    return saved, skipped_dup, skipped_wrong, skipped_lowconf

def download_and_process(query, expected_cat, max_videos=3):
    """Cari di YouTube, download, proses."""
    if new_counts[expected_cat] >= TARGET_NEW_PER_CAT:
        print(f"  [SKIP] {expected_cat} sudah cukup ({new_counts[expected_cat]}/{TARGET_NEW_PER_CAT})")
        return
    
    print(f"\n  [{expected_cat}] Mencari: \"{query}\"")
    
    ydl_opts = {
        'format': 'worstaudio/worst',
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'default_search': f'ytsearch{max_videos}',
        'max_downloads': max_videos,
        'ignoreerrors': True,
        'match_filter': yt_dlp.utils.match_filter_func(
            f'duration < {MAX_VIDEO_DURATION}'
        ),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts['outtmpl'] = os.path.join(tmpdir, '%(id)s.%(ext)s')
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch{max_videos}:{query}", download=True)
        except Exception as e:
            print(f"    [!] Error download: {e}")
            return
        
        # Proses semua file yang berhasil didownload
        wav_files = [f for f in os.listdir(tmpdir) if f.endswith('.wav')]
        for wf in wav_files:
            wpath = os.path.join(tmpdir, wf)
            s, sd, sw, sl = process_and_save(wpath, expected_cat, query)
            if s > 0:
                print(f"    [+] Disimpan: {s} chunk baru ke {expected_cat}/")
            if sd > 0:
                print(f"    [~] Dilewati duplikat: {sd}")
            if sw > 0:
                print(f"    [~] Dilewati salah kelas: {sw}")
            if sl > 0:
                print(f"    [~] Dilewati confidence rendah: {sl}")

# ─── MAIN LOOP ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MULAI PENGUMPULAN DATA OTOMATIS")
print("=" * 60)
print(f"Target: {TARGET_NEW_PER_CAT} file baru per kategori")
print(f"Threshold confidence: {SAVE_THRESHOLD*100:.0f}%\n")

for query, cat in SEARCH_QUERIES:
    download_and_process(query, cat, max_videos=3)
    time.sleep(1)  # Jeda sopan ke server YouTube

# ─── RINGKASAN ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SELESAI - RINGKASAN")
print("=" * 60)
total_new = 0
for cat in CATEGORIES:
    n = new_counts[cat]
    total_new += n
    print(f"  {cat:<12}: +{n:>4} file baru  (total: {existing_counts[cat]+n})")
print(f"\n  Total file baru: {total_new}")
print(f"  Tersimpan di  : {SAVE_BASE}")

if total_new > 0:
    print("\n[!] Data baru berhasil dikumpulkan!")
    print("    Sekarang jalankan: python train_lokal.py")
    print("    untuk melatih ulang AI dengan data yang lebih lengkap.")
else:
    print("\n[~] Tidak ada file baru yang memenuhi syarat.")
    print("    Coba turunkan SAVE_THRESHOLD atau tambah query pencarian.")
