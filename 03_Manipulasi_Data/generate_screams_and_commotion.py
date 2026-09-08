"""
generate_screams_and_commotion.py
==================================
Menghasilkan suara sintetis berteriak (screaming) dan keramaian/keributan (crowd babble)
untuk dimasukkan ke kelas NORMAL (class 2). Hal ini mencegah AI salah menebak saat ada
suara orang berteriak nada tinggi atau keributan di jalanan/dalam mobil.
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
print(" GENERATOR TERIAKAN (SCREAM) & KERIBUTAN (CROWD COMMOTION) SINTETIS")
print("=" * 80)

# Hapus file lama jika ada
import glob
for old_f in glob.glob(os.path.join(OUT_DIR, "scream_neg_*.wav")):
    try: os.remove(old_f)
    except: pass
for old_f in glob.glob(os.path.join(OUT_DIR, "crowd_neg_*.wav")):
    try: os.remove(old_f)
    except: pass

count = 0

# 1. GENERATOR TERIAKAN (Scream/Shriek) - 150 File
# Karakteristik: Frekuensi fundamental tinggi (600 - 1200 Hz), bergetar (vibrato/tremolo),
# dan mengandung derau (noise) nafas tinggi.
for i in range(150):
    # Pilih pitch dasar teriakan (nada tinggi)
    f0_base = np.random.uniform(600, 1000)
    
    # Getaran suara (unstable jitter / vibrato)
    vibrato_freq = np.random.uniform(6, 12) # Frekuensi getaran suara (Hz)
    vibrato_amp = np.random.uniform(20, 50)  # Rentang getaran pitch (Hz)
    jitter = np.random.normal(0, 10, num_samples) # Ketidakstabilan pita suara
    
    inst_freq = f0_base + vibrato_amp * np.sin(2 * np.pi * vibrato_freq * t) + jitter
    phase = 2 * np.pi * np.cumsum(inst_freq) / SR
    
    # Suara teriakan serak (distorsi harmonik vokal + derau udara)
    y = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.3 * np.sin(3 * phase)
    
    # Derau udara / hembusan angin (breathiness)
    breath_noise = np.random.normal(0, 0.3, num_samples)
    breath_noise = np.convolve(breath_noise, np.ones(3)/3, mode='same')
    
    y = y + breath_noise
    
    # Envelope teriakan (naik cepat di awal, bertahan keras, dan fade out di akhir)
    env = np.ones(num_samples)
    attack_samples = int(0.3 * SR)
    decay_samples = int(0.5 * SR)
    env[:attack_samples] = np.linspace(0, 1, attack_samples)
    env[-decay_samples:] = np.linspace(1, 0, decay_samples)
    
    # Modulasi volume mikro (tremolo suara bergetar karena emosi)
    tremolo = 1.0 + 0.2 * np.sin(2 * np.pi * np.random.uniform(5, 10) * t)
    y = y * env * tremolo
    
    # Normalisasi
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y /= max_val
    y = np.clip(y, -1.0, 1.0)
    y_int = (y * 32767).astype(np.int16)
    
    fn = os.path.join(OUT_DIR, f"scream_neg_synth_{i:03d}.wav")
    wav.write(fn, SR, y_int)
    count += 1

# 2. GENERATOR KERIBUTAN / CROWD COMMOTION (Babble Noise) - 150 File
# Karakteristik: Banyak suara orang bersahutan secara acak (overlaying voices).
# Menghasilkan spektrum padat (dense spectrum) yang menyatu.
for i in range(150):
    y = np.zeros(num_samples)
    num_speakers = np.random.randint(5, 10) # 5 s.d 10 suara bersahutan
    
    for _ in range(num_speakers):
        # Durasi & letak bicara tiap orang acak
        s_start = np.random.uniform(0, DURATION - 1.0)
        s_dur = np.random.uniform(0.5, 2.0)
        start_idx = int(s_start * SR)
        end_idx = int((s_start + s_dur) * SR)
        if end_idx >= num_samples: continue
        
        t_syl = np.arange(end_idx - start_idx) / SR
        f0 = np.random.uniform(150, 400) # Frekuensi ngobrol normal
        phase = 2 * np.pi * f0 * t_syl
        
        v_wave = np.sin(phase) + 0.5 * np.sin(2 * phase) + np.random.normal(0, 0.2, len(t_syl))
        env = np.sin(np.pi * (t_syl / s_dur)) ** 2
        
        y[start_idx:end_idx] += v_wave * env * np.random.uniform(0.3, 0.8)
        
    # Tambahkan kebisingan latar belakang jalan raya tipis
    y += np.random.normal(0, 0.02, num_samples)
    
    # Normalisasi
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y /= max_val
    y = np.clip(y, -1.0, 1.0)
    y_int = (y * 32767).astype(np.int16)
    
    fn = os.path.join(OUT_DIR, f"crowd_neg_synth_{i:03d}.wav")
    wav.write(fn, SR, y_int)
    count += 1

print(f"[OK] Berhasil membuat {count} file teriakan & keribut sintetis di folder NORMAL!")
print("=" * 80)
