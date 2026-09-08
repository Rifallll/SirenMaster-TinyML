"""
generate_synthetic_speech.py
============================
Menghasilkan suara percakapan/ngobrol manusia sintetis (robotic vocoder speech)
menggunakan modulasi formant vokal dan jeda suku kata. 
File disimpan ke kelas NORMAL (class 2) agar AI belajar menolak suara obrolan/podcast.
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
print(" GENERATOR PERCAKAPAN/OBROLAN MANUSIA SINTETIS (Vocoder Formant Speech)")
print("=" * 80)

# Hapus file speech_neg_*.wav lama jika ada
import glob
for old_f in glob.glob(os.path.join(OUT_DIR, "speech_neg_*.wav")):
    try: os.remove(old_f)
    except: pass

count = 0

for i in range(250): # Buat 250 file percakapan/obrolan sintetis
    y = np.zeros(num_samples)
    
    # Simulasikan suku kata (syllables) sepanjang 4 detik
    # 4 detik bisa memuat 10 s.d 15 suku kata
    num_syllables = np.random.randint(10, 16)
    syllable_starts = np.linspace(0.1, DURATION - 0.4, num_syllables)
    
    for start in syllable_starts:
        duration = np.random.uniform(0.15, 0.3) # Durasi satu suku kata (150ms - 300ms)
        start_idx = int(start * SR)
        end_idx = int((start + duration) * SR)
        if end_idx >= num_samples: break
        
        t_syl = np.arange(end_idx - start_idx) / SR
        
        # Frekuensi fundamental vokal manusia (F0)
        f0_start = np.random.uniform(120, 250)
        f0_end = f0_start + np.random.uniform(-40, 40)
        f0 = f0_start + (f0_end - f0_start) * (t_syl / duration)
        phase_f0 = 2 * np.pi * np.cumsum(f0) / SR
        
        # Formant vokal manusia:
        # F1 = ~500 Hz (resonansi tenggorokan)
        # F2 = ~1500 Hz (resonansi mulut)
        # F3 = ~2500 Hz (resonansi gigi/bibir)
        f1_mult = np.random.uniform(2.0, 3.5)
        f2_mult = np.random.uniform(5.5, 8.0)
        f3_mult = np.random.uniform(10.0, 14.0)
        
        # Gelombang vokal (vowel)
        vocal_wave = (
            np.sin(phase_f0) + 
            0.6 * np.sin(f1_mult * phase_f0) + 
            0.4 * np.sin(f2_mult * phase_f0) + 
            0.2 * np.sin(f3_mult * phase_f0)
        )
        
        # Konsonan geser (consonant noise) seperti /s/, /f/, /t/ di awal atau akhir suku kata
        noise = np.random.normal(0, 0.15, len(t_syl))
        # Filter bandpass sederhana untuk noise konsonan agar berada di 2kHz - 4kHz
        noise_filtered = np.convolve(noise, np.ones(5)/5, mode='same')
        
        # Gabungkan vokal dan konsonan
        mix_wave = vocal_wave.copy()
        if np.random.rand() > 0.4:
            # Tambahkan desis di awal (misal suara 's')
            noise_len = int(len(t_syl) * np.random.uniform(0.1, 0.3))
            mix_wave[:noise_len] += noise_filtered[:noise_len] * 0.8
        
        # Terapkan volume envelope (naik-turun tajam pada suku kata)
        envelope = np.sin(np.pi * (t_syl / duration)) ** 2
        mix_wave = mix_wave * envelope
        
        y[start_idx:end_idx] += mix_wave
        
    # Normalisasi volume
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y /= max_val
        
    # Tambahkan desis ambient ruangan tipis (white noise latar)
    y += np.random.normal(0, 0.005, num_samples)
    
    # Pastikan tetap dalam batas amplitude
    y = np.clip(y, -1.0, 1.0)
    y_int = (y * 32767).astype(np.int16)
    
    fn = os.path.join(OUT_DIR, f"speech_neg_synth_{i:03d}.wav")
    wav.write(fn, SR, y_int)
    count += 1

print(f"[OK] Berhasil membuat {count} file obrolan/podcast sintetis di folder NORMAL!")
print("=" * 80)
