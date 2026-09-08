"""
generate_wristwear_noise.py
============================
Menghasilkan suara sintetis khas perangkat wrist-worn (jam tangan) untuk kelas NORMAL:
1. Gesekan kain/baju ke mikrofon (cloth rustling)
2. Gerakan tangan memutar (arm swing air turbulence)
3. Ketukan jari ke permukaan (finger tapping)
4. Kontak kulit ke mikrofon (skin rubbing)

File-file ini mencegah AI salah mengira gesekan kain sebagai sirine saat pengguna bergerak.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, numpy as np, scipy.io.wavfile as wav
import glob

OUT_DIR = r"C:\Users\ASUS\Videos\DATASET\NORMAL"
os.makedirs(OUT_DIR, exist_ok=True)

SR = 8000
DURATION = 4.0
num_samples = int(SR * DURATION)
t = np.arange(num_samples) / SR

print("=" * 80)
print(" GENERATOR SUARA WRISTWEAR SINTETIS")
print(" (Gesekan Kain, Gerakan Tangan, Ketukan Jari, Kontak Kulit)")
print("=" * 80)

# Hapus file wristwear lama jika ada
for old_f in glob.glob(os.path.join(OUT_DIR, "wristwear_*.wav")):
    try: os.remove(old_f)
    except: pass

count = 0

# ─────────────────────────────────────────────────────────────────
# 1. GESEKAN KAIN KE MIC (Cloth Rustling)
# Karakteristik: Broadband noise (putih/coklat), amplitudo sangat fluktuatif & burst
# ─────────────────────────────────────────────────────────────────
print("\n[1/4] Generating cloth rustling sounds...")
for i in range(100):
    y = np.zeros(num_samples)

    # Buat 3-8 burst gesekan kain sepanjang 4 detik
    num_bursts = np.random.randint(3, 9)
    for _ in range(num_bursts):
        start = np.random.uniform(0, DURATION - 0.5)
        dur = np.random.uniform(0.08, 0.5) # Burst singkat 80ms-500ms
        s_idx = int(start * SR)
        e_idx = int((start + dur) * SR)
        if e_idx >= num_samples: continue

        burst_len = e_idx - s_idx
        # Broadband noise (putih) sebagai base kain
        noise = np.random.normal(0, 1.0, burst_len)

        # Warna noise mirip kain: filter lowpass ringan
        window = np.ones(4)/4
        noise = np.convolve(noise, window, mode='same')

        # Amplitude envelope yang tidak mulus (spiky / transient)
        env = np.random.uniform(0.2, 1.0, burst_len)
        env = np.convolve(env, np.ones(20)/20, mode='same') # Smoothing sedikit
        env = np.clip(env, 0, 1)

        y[s_idx:e_idx] += noise * env * np.random.uniform(0.3, 0.9)

    # Tambahkan sedikit desis mic
    y += np.random.normal(0, 0.01, num_samples)

    max_val = np.max(np.abs(y))
    if max_val > 0: y /= max_val
    y = np.clip(y, -1.0, 1.0)
    wav.write(os.path.join(OUT_DIR, f"wristwear_cloth_{i:03d}.wav"), SR, (y * 32767).astype(np.int16))
    count += 1

# ─────────────────────────────────────────────────────────────────
# 2. GERAKAN TANGAN / LENGAN (Arm Swing Air Turbulence)
# Karakteristik: Wind puff noise — low-frequency thumping + high freq noise
# ─────────────────────────────────────────────────────────────────
print("[2/4] Generating arm swing / air turbulence sounds...")
for i in range(80):
    y = np.zeros(num_samples)

    # Simulasikan 2-5 gerakan lengan
    num_swings = np.random.randint(2, 6)
    for _ in range(num_swings):
        start = np.random.uniform(0, DURATION - 0.8)
        dur = np.random.uniform(0.2, 0.8)
        s_idx = int(start * SR)
        e_idx = int((start + dur) * SR)
        if e_idx >= num_samples: continue

        t_swing = np.arange(e_idx - s_idx) / SR

        # Wind puff: low-freq bump (angin masuk) + decaying noise
        # Frekuensi sangat rendah 20-80 Hz (infrasonik)
        f_wind = np.random.uniform(20, 80)
        wind_base = np.sin(2 * np.pi * f_wind * t_swing)

        # Noise turbulens
        turb = np.random.normal(0, 0.5, len(t_swing))

        # Envelope: naik cepat, turun lebih lambat
        peak = int(len(t_swing) * 0.2)
        env = np.zeros(len(t_swing))
        env[:peak] = np.linspace(0, 1, peak)
        env[peak:] = np.linspace(1, 0.05, len(t_swing) - peak)

        y[s_idx:e_idx] += (wind_base * 0.5 + turb * 0.5) * env * np.random.uniform(0.4, 1.0)

    y += np.random.normal(0, 0.005, num_samples)
    max_val = np.max(np.abs(y))
    if max_val > 0: y /= max_val
    y = np.clip(y, -1.0, 1.0)
    wav.write(os.path.join(OUT_DIR, f"wristwear_swing_{i:03d}.wav"), SR, (y * 32767).astype(np.int16))
    count += 1

# ─────────────────────────────────────────────────────────────────
# 3. KETUKAN JARI / BENDA KE PERMUKAAN (Finger Tap / Surface Knock)
# Karakteristik: Transient impulsif singkat + decay
# ─────────────────────────────────────────────────────────────────
print("[3/4] Generating finger tap / surface knock sounds...")
for i in range(80):
    y = np.zeros(num_samples)

    # 4-12 ketukan acak
    num_taps = np.random.randint(4, 13)
    for _ in range(num_taps):
        start = np.random.uniform(0, DURATION - 0.1)
        s_idx = int(start * SR)

        # Panjang ketukan (sangat singkat: 30ms - 100ms)
        tap_len = int(np.random.uniform(0.03, 0.1) * SR)
        if s_idx + tap_len >= num_samples: continue

        t_tap = np.arange(tap_len) / SR

        # Suara ketukan: sinyal tinggi yang langsung decay eksponensial
        f_tap = np.random.uniform(200, 600)
        tap_sig = np.sin(2 * np.pi * f_tap * t_tap)
        decay = np.exp(-t_tap * np.random.uniform(30, 80)) # Decay cepat
        noise_tap = np.random.normal(0, 0.3, tap_len)

        y[s_idx:s_idx+tap_len] += (tap_sig * 0.7 + noise_tap * 0.3) * decay * np.random.uniform(0.5, 1.0)

    y += np.random.normal(0, 0.005, num_samples)
    max_val = np.max(np.abs(y))
    if max_val > 0: y /= max_val
    y = np.clip(y, -1.0, 1.0)
    wav.write(os.path.join(OUT_DIR, f"wristwear_tap_{i:03d}.wav"), SR, (y * 32767).astype(np.int16))
    count += 1

# ─────────────────────────────────────────────────────────────────
# 4. KONTAK KULIT KE MIC (Skin Contact / Muffled Mic)
# Karakteristik: Broadband noise sangat teredam (low-pass heavy) + low rumble
# ─────────────────────────────────────────────────────────────────
print("[4/4] Generating skin contact / muffled mic sounds...")
for i in range(60):
    y = np.zeros(num_samples)

    # Simulasikan momen mic tertutup kulit: broadband noise yang teredam frekuensi tinggi
    # dengan low-frequency rumble dari pompa darah/gerakan tulang
    start = np.random.uniform(0, DURATION - 1.5)
    dur = np.random.uniform(0.5, 2.0)
    s_idx = int(start * SR)
    e_idx = int((start + dur) * SR)
    if e_idx >= num_samples:
        e_idx = num_samples - 1

    contact_len = e_idx - s_idx

    # Noise teredam (kulit menyerap frekuensi tinggi)
    raw_noise = np.random.normal(0, 1.0, contact_len)

    # Heavy low-pass filter (kulit sangat menyerap HF)
    lp_window = np.ones(50)/50
    lp_noise = np.convolve(raw_noise, lp_window, mode='same')

    # Tambah low-frequency body rumble (25-60 Hz)
    t_contact = np.arange(contact_len) / SR
    f_body = np.random.uniform(25, 60)
    body_rumble = np.sin(2 * np.pi * f_body * t_contact) * 0.1

    y[s_idx:e_idx] += (lp_noise * 0.5 + body_rumble) * np.random.uniform(0.4, 0.8)

    # Sisa waktu: keheningan + desis tipis
    y += np.random.normal(0, 0.01, num_samples)

    max_val = np.max(np.abs(y))
    if max_val > 0: y /= max_val
    y = np.clip(y, -1.0, 1.0)
    wav.write(os.path.join(OUT_DIR, f"wristwear_skin_{i:03d}.wav"), SR, (y * 32767).astype(np.int16))
    count += 1

print(f"\n[OK] Berhasil membuat {count} file sintetis wristwear di folder NORMAL!")
print("  - cloth (gesekan kain)  : 100 file")
print("  - swing (gerakan lengan): 80 file")
print("  - tap   (ketukan jari)  : 80 file")
print("  - skin  (kontak kulit)  : 60 file")
print("=" * 80)
