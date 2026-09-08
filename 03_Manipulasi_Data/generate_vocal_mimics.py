"""
generate_vocal_mimics.py
========================
Membuat file audio WAV sintetis untuk mensimulasikan suara manusia (vokal/mulut/siulan)
yang meniru sirine ("ngiung-ngiung"). File ini disimpan ke kelas NORMAL (class 2)
agar model AI belajar menolak tiruan suara dari mulut.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, numpy as np, scipy.io.wavfile as wav

OUT_DIR = r"C:\Users\ASUS\Videos\DATASET\NORMAL"
os.makedirs(OUT_DIR, exist_ok=True)

SR = 8000
DURATION = 4.0
num_samples = int(SR * DURATION)
t = np.arange(num_samples) / SR

print("=" * 80)
print(" GENERATOR SINYAL TIRUAN VOKAL / MULUT (Mouth Mimicry)")
print("=" * 80)
print(f"Target folder: {OUT_DIR}")

# Hapus file vocal_mimic_*.wav lama jika ada agar bersih
import glob
for old_f in glob.glob(os.path.join(OUT_DIR, "vocal_mimic_*.wav")):
    try: os.remove(old_f)
    except: pass

count = 0

# 1. GENERATE WAIL MIMICS (Vokal / Siulan lambat naik-turun di frekuensi rendah vokal)
# Frekuensi fundamental F0 berkisar di 250 - 650 Hz (Vokal manusia)
for i in range(50):
    # Pilih parameter acak untuk variasi
    f_center = np.random.uniform(350, 500)
    f_dev = np.random.uniform(100, 150)
    wail_speed = np.random.uniform(0.2, 0.5) # Kecepatan lambat (naik turun dalam 2-5 detik)
    
    # Frekuensi instan
    inst_freq = f_center + f_dev * np.sin(2 * np.pi * wail_speed * t)
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    
    # Gelombang suara: vokal manusia didominasi oleh Fundamental + sedikit Harmonik ke-2
    # Tanpa harmonik tinggi (yang merupakan karakteristik sirine mekanik nyata)
    y = np.sin(phase) + 0.3 * np.sin(2 * phase) + np.random.normal(0, 0.01, num_samples)
    
    # Terapkan amplifikasi vokal (fade in/out pelan seperti nafas manusia)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 0.5 * t)) # Fluktuasi volume per nafasan
    y = y * envelope
    
    # Normalisasi
    y /= np.max(np.abs(y)) + 1e-9
    y = (y * 32767).astype(np.int16)
    
    fn = os.path.join(OUT_DIR, f"vocal_mimic_wail_{i:03d}.wav")
    wav.write(fn, SR, y)
    count += 1

# 2. GENERATE YELP MIMICS (Vokal cepat / Siulan yelp)
# Frekuensi fundamental 300 - 750 Hz, modulasi cepat (2-4 Hz)
for i in range(50):
    f_center = np.random.uniform(400, 550)
    f_dev = np.random.uniform(120, 180)
    yelp_speed = np.random.uniform(1.8, 3.5) # Cepat
    
    inst_freq = f_center + f_dev * np.sin(2 * np.pi * yelp_speed * t)
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    
    # Yelp mulut: dominan fundamental murni
    y = np.sin(phase) + np.random.normal(0, 0.005, num_samples)
    
    # Envelope
    envelope = np.random.uniform(0.7, 1.0)
    y = y * envelope
    
    # Normalisasi
    y /= np.max(np.abs(y)) + 1e-9
    y = (y * 32767).astype(np.int16)
    
    fn = os.path.join(OUT_DIR, f"vocal_mimic_yelp_{i:03d}.wav")
    wav.write(fn, SR, y)
    count += 1

# 3. GENERATE WHISTLE SWEEPS (Siulan naik terus / turun terus di frekuensi 400 - 900 Hz)
for i in range(50):
    f_start = np.random.uniform(350, 500)
    f_end = np.random.uniform(700, 950)
    if np.random.rand() > 0.5:
        f_start, f_end = f_end, f_start
        
    # Interpolasi frekuensi linear
    inst_freq = f_start + (f_end - f_start) * (t / DURATION)
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    
    y = np.sin(phase) + np.random.normal(0, 0.002, num_samples)
    
    # Normalisasi
    y /= np.max(np.abs(y)) + 1e-9
    y = (y * 32767).astype(np.int16)
    
    fn = os.path.join(OUT_DIR, f"vocal_mimic_sweep_{i:03d}.wav")
    wav.write(fn, SR, y)
    count += 1

print(f"[OK] Berhasil membuat {count} file sintetis 'vocal_mimic_*.wav' di folder NORMAL!")
print("=" * 80)
