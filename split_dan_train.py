"""
split_dan_train.py
==================
1. Potong file guru_normal_*.wav menjadi potongan 4 detik
2. Langsung jalankan training ulang model AI
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, glob, subprocess
import numpy as np

SAMPLE_RATE = 8000
DURATION    = 4.0  # detik per potongan
TARGET_LEN  = int(SAMPLE_RATE * DURATION)
NORMAL_DIR  = r"C:\Users\ASUS\Videos\DATASET\NORMAL"

try:
    import scipy.io.wavfile as wav
    import scipy.signal as signal
except ImportError:
    print("[!] Install scipy dulu: pip install scipy"); sys.exit(1)

# ── STEP 1: Potong file guru_normal_*.wav ────────────────────────────
print("="*60)
print(" STEP 1: Memotong file rekaman NORMAL menjadi 4 detik...")
print("="*60)

guru_files = glob.glob(os.path.join(NORMAL_DIR, "guru_normal_*.wav"))
total_potongan = 0

for fpath in guru_files:
    print(f"\n[*] Memproses: {os.path.basename(fpath)}")
    try:
        sr, data = wav.read(fpath)
        # Konversi ke float32
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        
        # Resample jika perlu
        if sr != SAMPLE_RATE:
            print(f"    Resample dari {sr}Hz ke {SAMPLE_RATE}Hz...")
            num_samples = int(len(data) * SAMPLE_RATE / sr)
            data = signal.resample(data, num_samples)
        
        # Potong jadi segmen 4 detik
        n_segments = len(data) // TARGET_LEN
        print(f"    Durasi: {len(data)/SAMPLE_RATE:.1f} detik → {n_segments} potongan 4 detik")
        
        base_name = os.path.splitext(os.path.basename(fpath))[0]
        for i in range(n_segments):
            segment = data[i*TARGET_LEN:(i+1)*TARGET_LEN]
            # Normalisasi volume segmen
            max_val = np.max(np.abs(segment))
            if max_val > 1e-6:
                segment = segment * (0.90 / max_val)
            # Simpan
            out_name = f"split_{base_name}_seg{i+1:03d}.wav"
            out_path = os.path.join(NORMAL_DIR, out_name)
            wav_data = np.clip(segment * 32767, -32768, 32767).astype(np.int16)
            wav.write(out_path, SAMPLE_RATE, wav_data)
            total_potongan += 1
        
        print(f"    ✅ {n_segments} file tersimpan!")
    except Exception as e:
        print(f"    [!] Error: {e}")

print(f"\n[SELESAI] Total {total_potongan} potongan baru di folder NORMAL")

if total_potongan == 0:
    print("[!] Tidak ada potongan yang dihasilkan. Periksa file rekaman Anda.")
    sys.exit(1)

# ── STEP 2: Jalankan Training ────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 2: Mulai Training Model AI Baru...")
print(" (Ini akan memakan waktu 20-40 menit, harap tunggu!)")
print("="*60)

train_script = r"C:\Users\ASUS\Videos\DATASET\04_Training_AI\train_lokal.py"
if not os.path.exists(train_script):
    print(f"[!] Script training tidak ditemukan: {train_script}")
    sys.exit(1)

result = subprocess.run(
    ["python", train_script],
    cwd=r"C:\Users\ASUS\Videos\DATASET\04_Training_AI"
)

if result.returncode == 0:
    print("\n" + "="*60)
    print(" ✅ TRAINING SELESAI!")
    print(" Model baru sudah tersimpan di:")
    print("   C:\\Users\\ASUS\\Videos\\DATASET\\sirenmaster_main\\model.h")
    print(" Silakan Upload ulang ke ESP32!")
    print("="*60)
else:
    print("\n[!] Training gagal. Periksa pesan error di atas.")
