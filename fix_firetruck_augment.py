"""
fix_firetruck_augment.py
========================
Membuat augmentasi FIRETRUCK yang lebih beragam agar model tidak bingung
membedakan sirine damkar vs ambulans.

Strategi:
1. Pitch-shift 10 step dari -4 ke +4 semitones
2. Noise injection: campur dgn background traffic ringan
3. Time-stretch (speed up/slow down)
4. Volume variation

Target: 400+ file augmentasi baru agar setara dengan AMBULANCE
"""

import os, sys, glob, random
import numpy as np
import librosa
import soundfile as sf

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.random.seed(42)
random.seed(42)

FIRETRUCK_DIR = r"C:\Users\ASUS\Videos\DATASET\FIRETRUCK"
NORMAL_DIR = r"C:\Users\ASUS\Videos\DATASET\NORMAL"
SR = 8000
DUR = 4.0
TARGET_LEN = int(SR * DUR)

# --- Ambil file FIRETRUCK asli yang bagus (bukan augmentasi) ---
orig_files = sorted(glob.glob(os.path.join(FIRETRUCK_DIR, "fire_*.wav")))
synth_files = glob.glob(os.path.join(FIRETRUCK_DIR, "SYNTH_*.wav"))
guru_files = glob.glob(os.path.join(FIRETRUCK_DIR, "guru_*.wav"))
dl_files = glob.glob(os.path.join(FIRETRUCK_DIR, "dl_*.wav"))

# Kombinasikan semua sumber asli
source_files = orig_files + synth_files + guru_files + dl_files
random.shuffle(source_files)
print(f"Sumber file FIRETRUCK asli: {len(source_files)} file")
print(f"  fire_: {len(orig_files)}, SYNTH: {len(synth_files)}, guru: {len(guru_files)}, dl_: {len(dl_files)}")

# --- Load background noise ringan ---
noise_candidates = (
    glob.glob(os.path.join(NORMAL_DIR, "traffic_*.wav"))[:30] +
    glob.glob(os.path.join(NORMAL_DIR, "motor_*.wav"))[:20] +
    glob.glob(os.path.join(NORMAL_DIR, "aug_noise_*.wav"))[:50]
)
random.shuffle(noise_candidates)
print(f"Background noise kandidat: {len(noise_candidates)} file")

def load_and_pad(filepath, sr=SR, dur=DUR):
    try:
        y, _ = librosa.load(filepath, sr=sr, mono=True)
        tl = int(sr * dur)
        if len(y) < tl:
            y = np.pad(y, (0, tl - len(y)))
        else:
            y = y[:tl]
        return y
    except:
        return None

def normalize(y, target=0.7):
    mx = np.max(np.abs(y))
    if mx > 1e-6:
        return y * (target / mx)
    return y

# --- Cek berapa augmentasi yang sudah ada ---
exist_aug_pitch = glob.glob(os.path.join(FIRETRUCK_DIR, "aug_pitch_*.wav"))
exist_aug_noise = glob.glob(os.path.join(FIRETRUCK_DIR, "aug_noise_*.wav"))
exist_aug_stretch = glob.glob(os.path.join(FIRETRUCK_DIR, "aug_stretch_*.wav"))
print(f"\nAugmentasi yang sudah ada: pitch={len(exist_aug_pitch)}, noise={len(exist_aug_stretch)}")

# Target: buat total ~400 augmentasi baru
N_PER_AUG = 4  # augmentasi per file sumber
sources_to_use = source_files[:100]  # pakai 100 file sumber terbaik

print(f"\n[*] Membuat augmentasi FIRETRUCK baru...")
created = 0

for i, src_path in enumerate(sources_to_use):
    y = load_and_pad(src_path)
    if y is None:
        continue
    y = normalize(y)
    base = f"aug_fix_{i:04d}"

    # 1. Pitch shift -3 semitones (bikin lebih rendah/berat)
    try:
        y_low = librosa.effects.pitch_shift(y, sr=SR, n_steps=-3)
        out_path = os.path.join(FIRETRUCK_DIR, f"{base}_pitch_low.wav")
        sf.write(out_path, normalize(y_low), SR)
        created += 1
    except: pass

    # 2. Pitch shift +3 semitones (bikin lebih tinggi/kencang)
    try:
        y_high = librosa.effects.pitch_shift(y, sr=SR, n_steps=3)
        out_path = os.path.join(FIRETRUCK_DIR, f"{base}_pitch_high.wav")
        sf.write(out_path, normalize(y_high), SR)
        created += 1
    except: pass

    # 3. Time-stretch 0.9x (diperlambat)
    try:
        y_slow = librosa.effects.time_stretch(y, rate=0.9)
        tl = int(SR * DUR)
        y_slow = y_slow[:tl] if len(y_slow) >= tl else np.pad(y_slow, (0, tl - len(y_slow)))
        out_path = os.path.join(FIRETRUCK_DIR, f"{base}_slow.wav")
        sf.write(out_path, normalize(y_slow), SR)
        created += 1
    except: pass

    # 4. Noise injection: campur traffic ringan (SNR ~15dB)
    if noise_candidates:
        nc = random.choice(noise_candidates)
        yn = load_and_pad(nc)
        if yn is not None:
            try:
                siren_rms = np.sqrt(np.mean(y**2)) + 1e-9
                noise_rms = np.sqrt(np.mean(yn**2)) + 1e-9
                snr_target = 10 ** (15/20)  # 15 dB SNR
                scale = siren_rms / (noise_rms * snr_target)
                y_mix = normalize(y + yn * scale * 0.3)
                out_path = os.path.join(FIRETRUCK_DIR, f"{base}_traffic.wav")
                sf.write(out_path, y_mix, SR)
                created += 1
            except: pass

    if (i+1) % 20 == 0:
        print(f"  Progress: {i+1}/{len(sources_to_use)} sumber, {created} file dibuat...")

print(f"\n[OK] Selesai! {created} file augmentasi FIRETRUCK baru berhasil dibuat.")
print(f"Total FIRETRUCK sekarang: {len(glob.glob(os.path.join(FIRETRUCK_DIR, '*.wav')))}")
