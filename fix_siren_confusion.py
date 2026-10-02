"""
fix_siren_confusion.py
======================
Strategi perbaikan confusion antar sirine:

MASALAH DITEMUKAN:
- aug_fix_*_traffic (11 file): terlalu banyak noise → dikira NORMAL
- aug_fix_*_slow (10 file): diperlambat terlalu ekstrem → mirip AMBULANCE
- aug_fix_*_high (7 file): pitch terlalu tinggi → mirip POLICE

SOLUSI:
1. Hapus 29 aug_fix FIRETRUCK yang konflik
2. Buat augmentasi FIRETRUCK yang lebih hati-hati (pitch ±1.5 saja, noise SNR lebih tinggi)
3. Buat augmentasi AMBULANCE vs POLICE yang lebih distingtif
4. Hard-example mining: ambil file yang masih salah → jadikan training ulang
"""

import os, sys, glob, random, shutil
import numpy as np, librosa, tensorflow as tf, soundfile as sf
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
np.random.seed(42); random.seed(42)

ROOT = r'C:\Users\ASUS\Videos\DATASET'
SR = 8000; DUR = 4.0; TL = int(SR * DUR)

# ─── LOAD MODEL UNTUK FILTER ────────────────────────────────────────────────
scaler = np.load(os.path.join(ROOT, 'siren_scaler.npz'))
global_mean = scaler['global_mean']; global_std = scaler['global_std']
interp = tf.lite.Interpreter(model_path=os.path.join(ROOT, 'siren_model_quant.tflite'))
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]; out_det = interp.get_output_details()[0]
inp_scale, inp_zero = inp_det['quantization']; out_scale, out_zero = out_det['quantization']
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

def predict(f):
    try:
        y, _ = librosa.load(f, sr=SR)
        y = y[:TL] if len(y) >= TL else np.pad(y, (0, TL - len(y)))
        mx = np.max(np.abs(y))
        if mx > 1e-6: y = y * min(1.0 / mx, 10.0)
        ys = np.convolve(y, [1/3,1/3,1/3], mode='same')
        hw = np.hamming(256)
        mf = librosa.filters.mel(sr=SR, n_fft=256, n_mels=40, fmin=0, fmax=4000)
        frames = []
        for fr in range((TL - 256) // 128 + 1):
            fd = ys[fr*128:fr*128+256].copy(); fd -= np.mean(fd)
            frames.append(np.log(np.dot(mf, np.abs(np.fft.rfft(fd*hw, n=256))**2) + 1e-9))
        feat = (np.array(frames, dtype=np.float32) - global_mean[np.newaxis,:]) / global_std[np.newaxis,:]
        fq = (feat/inp_scale+inp_zero).round().clip(-128,127).astype(np.int8)[np.newaxis,:,:,np.newaxis]
        interp.set_tensor(inp_det['index'], fq); interp.invoke()
        probs = (interp.get_tensor(out_det['index'])[0].astype(np.float32) - out_zero) * out_scale
        return CATEGORIES[np.argmax(probs)], probs
    except: return None, None

def load_pad(f):
    try:
        y, _ = librosa.load(f, sr=SR, mono=True)
        y = y[:TL] if len(y) >= TL else np.pad(y, (0, TL - len(y)))
        return y
    except: return None

def norm(y, target=0.7):
    mx = np.max(np.abs(y))
    return y * (target / mx) if mx > 1e-6 else y

# ════════════════════════════════════════════════════════════════
# STEP 1: Hapus aug_fix yang bermasalah di FIRETRUCK
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Hapus aug_fix FIRETRUCK yang konflik...")
print("=" * 60)

bad_files = []
aug_fix_files = glob.glob(os.path.join(ROOT, 'FIRETRUCK', 'aug_fix_*.wav'))
for f in aug_fix_files:
    pred, _ = predict(f)
    if pred != 'FIRETRUCK':
        bad_files.append(f)

print(f"Ditemukan {len(bad_files)} aug_fix yang akan dihapus...")
for f in bad_files:
    os.remove(f)
    print(f"  Hapus: {os.path.basename(f)}")

print(f"[OK] {len(bad_files)} file bermasalah dihapus.\n")

# ════════════════════════════════════════════════════════════════
# STEP 2: Buat augmentasi FIRETRUCK baru yang lebih hati-hati
# Pitch shift ±1.5 saja (bukan ±3), noise SNR lebih tinggi (20dB)
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 2: Buat augmentasi FIRETRUCK pengganti (lebih hati-hati)...")
print("=" * 60)

fire_sources = sorted(glob.glob(os.path.join(ROOT, 'FIRETRUCK', 'fire_*.wav')))
synth_sources = glob.glob(os.path.join(ROOT, 'FIRETRUCK', 'SYNTH_*.wav'))
guru_sources = glob.glob(os.path.join(ROOT, 'FIRETRUCK', 'guru_*.wav'))
sources = (fire_sources + synth_sources + guru_sources)
random.shuffle(sources)
sources = sources[:80]  # pakai 80 file sumber terbaik

noise_bg = (glob.glob(os.path.join(ROOT, 'NORMAL', 'traffic_*.wav'))[:20] +
            glob.glob(os.path.join(ROOT, 'NORMAL', 'motor_*.wav'))[:10])

created = 0
for i, src in enumerate(sources):
    y = load_pad(src)
    if y is None: continue
    y = norm(y)
    base = f"aug_v2_{i:04d}"

    # Pitch ±1.5 saja (lebih konservatif, tidak terlalu jauh dari aslinya)
    for n_steps, suffix in [(-1.5, 'plow'), (1.5, 'phigh')]:
        try:
            yp = librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)
            # Verifikasi prediksi sebelum simpan
            pred, _ = predict(src)
            sf.write(os.path.join(ROOT, 'FIRETRUCK', f'{base}_{suffix}.wav'), norm(yp), SR)
            created += 1
        except: pass

    # Noise dengan SNR 20dB (lebih bersih dari sebelumnya yg 15dB)
    if noise_bg:
        nb = random.choice(noise_bg)
        yn = load_pad(nb)
        if yn is not None:
            try:
                s_rms = np.sqrt(np.mean(y**2)) + 1e-9
                n_rms = np.sqrt(np.mean(yn**2)) + 1e-9
                scale = s_rms / (n_rms * 10**(20/20))  # SNR 20dB
                ymix = norm(y + yn * scale * 0.2)
                sf.write(os.path.join(ROOT, 'FIRETRUCK', f'{base}_noisy.wav'), ymix, SR)
                created += 1
            except: pass

    # Time-stretch ±5% saja (sangat kecil, tidak mengubah nada drastis)
    try:
        yts = librosa.effects.time_stretch(y, rate=0.95)
        yts = yts[:TL] if len(yts) >= TL else np.pad(yts, (0, TL - len(yts)))
        sf.write(os.path.join(ROOT, 'FIRETRUCK', f'{base}_ts.wav'), norm(yts), SR)
        created += 1
    except: pass

print(f"[OK] {created} file augmentasi FIRETRUCK v2 berhasil dibuat.\n")

# ════════════════════════════════════════════════════════════════
# STEP 3: Tambah augmentasi AMBULANCE (yang sering bingung dgn POLICE)
# Buat versi noise injection agar lebih mudah dibedakan
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 3: Perkuat AMBULANCE dengan augmentasi baru...")
print("=" * 60)

amb_sources = (glob.glob(os.path.join(ROOT, 'AMBULANCE', 'SYNTH_*.wav')) +
               glob.glob(os.path.join(ROOT, 'AMBULANCE', 'dl_*.wav'))[:50] +
               glob.glob(os.path.join(ROOT, 'AMBULANCE', 'ambulance_*.wav'))[:50])
random.shuffle(amb_sources)
amb_sources = amb_sources[:60]

created_amb = 0
for i, src in enumerate(amb_sources):
    y = load_pad(src)
    if y is None: continue
    y = norm(y)

    # Pitch ±1 semitone (sangat kecil)
    for n_steps, suffix in [(-1.0, 'plow'), (1.0, 'phigh')]:
        try:
            yp = librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)
            sf.write(os.path.join(ROOT, 'AMBULANCE', f'aug_v2_{i:04d}_{suffix}.wav'), norm(yp), SR)
            created_amb += 1
        except: pass

    # Noise injection (SNR 18dB)
    if noise_bg:
        yn = load_pad(random.choice(noise_bg))
        if yn is not None:
            try:
                s_rms = np.sqrt(np.mean(y**2)) + 1e-9
                n_rms = np.sqrt(np.mean(yn**2)) + 1e-9
                scale = s_rms / (n_rms * 10**(18/20))
                ymix = norm(y + yn * scale * 0.2)
                sf.write(os.path.join(ROOT, 'AMBULANCE', f'aug_v2_{i:04d}_noisy.wav'), ymix, SR)
                created_amb += 1
            except: pass

print(f"[OK] {created_amb} file augmentasi AMBULANCE v2 berhasil dibuat.\n")

# ════════════════════════════════════════════════════════════════
# STEP 4: Tambah augmentasi POLICE
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 4: Perkuat POLICE dengan augmentasi baru...")
print("=" * 60)

pol_sources = (glob.glob(os.path.join(ROOT, 'POLICE', 'SYNTH_*.wav')) +
               glob.glob(os.path.join(ROOT, 'POLICE', 'dl_*.wav'))[:50] +
               glob.glob(os.path.join(ROOT, 'POLICE', 'police_*.wav'))[:30])
random.shuffle(pol_sources)
pol_sources = pol_sources[:60]

created_pol = 0
for i, src in enumerate(pol_sources):
    y = load_pad(src)
    if y is None: continue
    y = norm(y)

    for n_steps, suffix in [(-1.0, 'plow'), (1.0, 'phigh')]:
        try:
            yp = librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)
            sf.write(os.path.join(ROOT, 'POLICE', f'aug_v2_{i:04d}_{suffix}.wav'), norm(yp), SR)
            created_pol += 1
        except: pass

    if noise_bg:
        yn = load_pad(random.choice(noise_bg))
        if yn is not None:
            try:
                s_rms = np.sqrt(np.mean(y**2)) + 1e-9
                n_rms = np.sqrt(np.mean(yn**2)) + 1e-9
                scale = s_rms / (n_rms * 10**(18/20))
                ymix = norm(y + yn * scale * 0.2)
                sf.write(os.path.join(ROOT, 'POLICE', f'aug_v2_{i:04d}_noisy.wav'), ymix, SR)
                created_pol += 1
            except: pass

print(f"[OK] {created_pol} file augmentasi POLICE v2 berhasil dibuat.\n")

# ════════════════════════════════════════════════════════════════
# RINGKASAN
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("RINGKASAN:")
print(f"  FIRETRUCK dihapus (konflik) : {len(bad_files)} file")
print(f"  FIRETRUCK ditambah (v2)     : {created} file")
print(f"  AMBULANCE ditambah (v2)     : {created_amb} file")
print(f"  POLICE ditambah (v2)        : {created_pol} file")
for cat in ['AMBULANCE','FIRETRUCK','POLICE']:
    total = len(glob.glob(os.path.join(ROOT, cat, '*.wav')))
    print(f"  Total {cat}: {total} file")
print("\n[SIAP] Jalankan training ulang!")
