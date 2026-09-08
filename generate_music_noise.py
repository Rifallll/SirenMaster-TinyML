"""
generate_music_noise.py
=======================
Generate audio sintetis mirip musik (rock, drum, bass, distorsi gitar)
Tambahkan ke folder NORMAL, lalu langsung retrain model AI.

Tujuan: Mengajarkan AI bahwa suara musik ≠ sirine darurat.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, time, subprocess
import numpy as np
import scipy.io.wavfile as wav

SAMPLE_RATE = 8000
DURATION    = 4.0
N_SAMPLES   = int(SAMPLE_RATE * DURATION)
NORMAL_DIR  = r"C:\Users\ASUS\Videos\DATASET\NORMAL"
N_GENERATE  = 200  # Jumlah file yang digenerate (bisa dinaikkan)

os.makedirs(NORMAL_DIR, exist_ok=True)

def save_wav(data, filename):
    """Simpan array float32 ke WAV 16-bit"""
    # Normalisasi ke 90% agar tidak clipping
    mx = np.max(np.abs(data))
    if mx > 1e-6:
        data = data * (0.90 / mx)
    out = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    wav.write(filename, SAMPLE_RATE, out)

def distort(sig, gain=8.0):
    """Soft-clip distorsi seperti efek gitar overdrive"""
    sig = sig * gain
    return np.tanh(sig)

def gen_guitar_chord(t, root_hz, n_strings=6):
    """Chord gitar dengan harmonik"""
    ratios = [1.0, 1.26, 1.50, 2.0, 2.52, 3.0][:n_strings]
    sig = np.zeros_like(t)
    for r in ratios:
        freq = root_hz * r
        amp  = 1.0 / (r + 0.5)
        sig += amp * np.sin(2 * np.pi * freq * t)
    return distort(sig, gain=np.random.uniform(4, 12))

def gen_drum_kick(t):
    """Suara kick drum (low thump)"""
    freq = 60 * np.exp(-20 * t)
    env  = np.exp(-15 * t)
    return env * np.sin(2 * np.pi * freq * t)

def gen_drum_snare(t):
    """Suara snare (gabungan tone + noise)"""
    tone  = 0.5 * np.exp(-30 * t) * np.sin(2 * np.pi * 200 * t)
    noise = 0.5 * np.exp(-20 * t) * np.random.randn(len(t))
    return tone + noise

def gen_hihat(t):
    """Hi-hat (high freq noise burst)"""
    return np.exp(-80 * t) * np.random.randn(len(t))

def gen_bass(t, root_hz):
    """Bass guitar - frekuensi rendah"""
    sig = np.sin(2 * np.pi * root_hz * t)
    sig += 0.3 * np.sin(2 * np.pi * root_hz * 2 * t)
    return distort(sig, gain=3.0)

def add_rhythm(base_signal, bpm, pattern_fn, sr=SAMPLE_RATE):
    """Tambahkan pola ritme ke sinyal"""
    beat_len = int(sr * 60 / bpm)
    result   = base_signal.copy()
    pos = 0
    while pos < len(result):
        burst = pattern_fn(np.linspace(0, 0.1, min(int(sr * 0.1), len(result) - pos)))
        end = min(pos + len(burst), len(result))
        result[pos:end] += burst[:end - pos] * 0.4
        pos += beat_len
    return result

def generate_rock_sample(seed):
    np.random.seed(seed)
    t   = np.linspace(0, DURATION, N_SAMPLES)
    sig = np.zeros(N_SAMPLES)

    # Pilih root note secara acak (A2=110Hz ~ E4=330Hz)
    root = np.random.choice([110, 123, 130, 147, 165, 185, 196, 220, 247, 261, 294, 330])
    bpm  = np.random.randint(80, 180)

    # Layer 1: Gitar rhythm (berulang setiap beat)
    beat_samples = int(SAMPLE_RATE * 60 / bpm)
    chord_dur    = np.random.choice([0.25, 0.5, 1.0])
    chord_len    = int(SAMPLE_RATE * chord_dur)
    pos = 0
    while pos < N_SAMPLES:
        end     = min(pos + chord_len, N_SAMPLES)
        t_chunk = np.linspace(0, chord_dur, end - pos)
        # Variasikan chord (root, fourth, fifth)
        freq = root * np.random.choice([1.0, 1.33, 1.5, 0.75, 2.0])
        chunk = gen_guitar_chord(t_chunk, freq)
        # Envelope: attack + decay
        env = np.linspace(0, 1, min(100, len(chunk)))
        env = np.concatenate([env, np.ones(len(chunk) - len(env))])
        sig[pos:end] += chunk * env * np.random.uniform(0.3, 0.8)
        pos += beat_len + np.random.randint(-10, 10)

    # Layer 2: Bass line
    bass_sig = gen_bass(t, root / 2)
    # Modulasi amplitude bass sesuai beat
    for i in range(0, N_SAMPLES, beat_samples):
        end = min(i + beat_samples // 2, N_SAMPLES)
        sig[i:end] += bass_sig[i:end] * 0.4

    # Layer 3: Drum kit
    for i in range(0, N_SAMPLES, beat_samples):
        # Kick setiap beat
        klen = min(int(SAMPLE_RATE * 0.15), N_SAMPLES - i)
        if klen > 0:
            tk = np.linspace(0, 0.15, klen)
            sig[i:i+klen] += gen_drum_kick(tk) * 0.6

        # Snare setiap 2 beat
        if (i // beat_samples) % 2 == 1:
            slen = min(int(SAMPLE_RATE * 0.1), N_SAMPLES - i)
            if slen > 0:
                ts = np.linspace(0, 0.1, slen)
                sig[i:i+slen] += gen_drum_snare(ts) * 0.5

        # Hi-hat setiap setengah beat
        half = beat_samples // 2
        for offset in [0, half]:
            pos = i + offset
            hlen = min(int(SAMPLE_RATE * 0.05), N_SAMPLES - pos)
            if hlen > 0 and pos < N_SAMPLES:
                th = np.linspace(0, 0.05, hlen)
                sig[pos:pos+hlen] += gen_hihat(th) * 0.25

    # Layer 4: Reverb sederhana (echo pendek)
    delay = int(SAMPLE_RATE * 0.08)
    echo  = np.zeros_like(sig)
    echo[delay:] = sig[:-delay] * 0.25
    sig += echo

    return sig.astype(np.float32)

def generate_variations(seed):
    """Generate berbagai jenis noise musik"""
    np.random.seed(seed)
    t = np.linspace(0, DURATION, N_SAMPLES)
    kind = seed % 5

    if kind == 0:
        # Pure rock
        return generate_rock_sample(seed)
    elif kind == 1:
        # Metal - lebih cepat, lebih distorsi
        sig = generate_rock_sample(seed + 1000)
        return distort(sig, gain=np.random.uniform(2, 5))
    elif kind == 2:
        # Musik dengan vokal sintetis (suara manusia + musik)
        sig  = generate_rock_sample(seed + 2000) * 0.6
        # Tambah suara vokal (glottal pulse model)
        f0   = np.random.uniform(80, 250)  # frekuensi dasar vokal
        vokal = np.sin(2 * np.pi * f0 * t)
        for h in range(2, 8):
            vokal += (1/h) * np.sin(2 * np.pi * f0 * h * t + np.random.uniform(0, np.pi))
        # Modulasi amplitudo seperti nyanyian
        mod  = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        sig += vokal * mod * 0.4
        return sig.astype(np.float32)
    elif kind == 3:
        # TV/Radio noise (campuran suara acak berfrekuensi tinggi)
        noise = np.random.randn(N_SAMPLES)
        # Band-pass filter kasar
        freqs = np.fft.rfft(noise)
        f = np.fft.rfftfreq(N_SAMPLES, 1/SAMPLE_RATE)
        mask = (f > 200) & (f < 3000)
        freqs[~mask] *= 0.1
        sig = np.fft.irfft(freqs, N_SAMPLES).astype(np.float32)
        return sig
    else:
        # Drum-only (tanpa melodi)
        sig = np.zeros(N_SAMPLES)
        bpm = np.random.randint(90, 200)
        beat = int(SAMPLE_RATE * 60 / bpm)
        for i in range(0, N_SAMPLES, beat):
            kl = min(int(SAMPLE_RATE*0.12), N_SAMPLES-i)
            if kl > 0:
                sig[i:i+kl] += gen_drum_kick(np.linspace(0,0.12,kl)) * 0.7
            sl = min(int(SAMPLE_RATE*0.08), N_SAMPLES-i)
            if sl > 0 and (i//beat)%2==1:
                sig[i:i+sl] += gen_drum_snare(np.linspace(0,0.08,sl)) * 0.5
        return sig.astype(np.float32)

# ── MAIN ─────────────────────────────────────────────────────────────
print("="*60)
print(" GENERATOR AUDIO MUSIK SINTETIS UNTUK DATASET NORMAL")
print("="*60)
print(f" Target: {N_GENERATE} file audio mirip musik/rock")
print(f" Folder: {NORMAL_DIR}")
print()

generated = 0
ts = int(time.time())

for i in range(N_GENERATE):
    try:
        sig = generate_variations(i + ts)
        fname = os.path.join(NORMAL_DIR, f"music_synth_{ts}_{i:04d}.wav")
        save_wav(sig, fname)
        generated += 1
        if (i+1) % 20 == 0:
            print(f" [{i+1}/{N_GENERATE}] Generated {generated} file...")
    except Exception as e:
        print(f" [!] Error seed {i}: {e}")

print(f"\n [SELESAI] {generated} file musik sintetis tersimpan di NORMAL!")
print()
print("="*60)
print(" Memulai Retrain Model AI...")
print(" (Estimasi: 30-40 menit)")
print("="*60)

train_script = r"C:\Users\ASUS\Videos\DATASET\04_Training_AI\train_lokal.py"
result = subprocess.run(
    ["python", train_script],
    cwd=r"C:\Users\ASUS\Videos\DATASET\04_Training_AI"
)

if result.returncode == 0:
    print("\n" + "="*60)
    print(" SELESAI! Model baru tersimpan di sirenmaster_main/model.h")
    print(" Upload ke ESP32 dan musik rock tidak akan deteksi lagi!")
    print("="*60)
else:
    print("\n[!] Training gagal. Cek error di atas.")
