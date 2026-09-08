"""
clean_dataset_v2.py
===================
Bersihkan dataset secara otomatis sebelum retrain.

Kriteria file yang dipindahkan ke TRASH:
1. File yang skor kelas sendiri < 15% DAN skor kelas lain > 80% (sangat mislabel)
2. File SYNTH_fire_* yang diprediksi sebagai AMBULANCE (sintesis salah)
3. File AMBULANCE v2_* yang diprediksi sebagai NORMAL/NOISE (rekaman buruk)
4. 18 file POLICE yang ambigu (sudah diidentifikasi dari audit)

Semua file dipindahkan ke TRASH/ (tidak dihapus permanen).
"""

import os, sys, re, glob, shutil, numpy as np, librosa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Config ──────────────────────────────────────────────────────────
BASE_DIR    = r"C:\Users\ASUS\Videos\DATASET"
TRASH_DIR   = os.path.join(BASE_DIR, "TRASH", "cleaned_v2")
MODEL_PATH  = os.path.join(BASE_DIR, "siren_model_quant.tflite")
MODEL_H     = os.path.join(BASE_DIR, "sirenmaster_main", "model.h")
CATEGORIES  = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
SAMPLE_RATE = 8000
DURATION    = 4.0
TARGET_LEN  = int(SAMPLE_RATE * DURATION)
os.makedirs(TRASH_DIR, exist_ok=True)

# ── Load scaler ──────────────────────────────────────────────────────
with open(MODEL_H, 'r') as f:
    content = f.read()
mean_m = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\}', content)
std_m  = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\}',  content)
MEL_MEAN = np.array([float(x.strip().rstrip('f')) for x in mean_m.group(1).split(',')])
MEL_STD  = np.array([float(x.strip().rstrip('f')) for x in std_m.group(1).split(',')])

def extract_features(y):
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    else:
        y = y[:TARGET_LEN]
    mx = np.max(np.abs(y))
    if mx > 1e-6:
        y = y * min(1.0 / mx, 10.0)
    y = np.convolve(y, [1/3, 1/3, 1/3], mode='same')
    n_frames = (TARGET_LEN - 256) // 128 + 1
    hamming  = np.hamming(256)
    mel_fb   = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=256, n_mels=40, fmin=0, fmax=4000)
    frames   = []
    for i in range(n_frames):
        st  = i * 128
        fd  = y[st:st+256].copy()
        fd -= np.mean(fd)
        pwr = np.abs(np.fft.rfft(fd * hamming, n=256)) ** 2
        frames.append(np.log(mel_fb @ pwr[:129] + 1e-9))
    spec = np.array(frames, dtype=np.float32)
    return (spec - MEL_MEAN) / (MEL_STD + 1e-8)

# ── TFLite interpreter ───────────────────────────────────────────────
interp = tf.lite.Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
inp_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]

def predict(spec):
    inp = spec[np.newaxis, :, :, np.newaxis].astype(np.float32)
    if inp_d['dtype'] == np.int8:
        sc = inp_d['quantization_parameters']['scales'][0]
        zp = inp_d['quantization_parameters']['zero_points'][0]
        inp_q = np.clip(np.round(inp / sc) + zp, -128, 127).astype(np.int8)
        interp.set_tensor(inp_d['index'], inp_q)
        interp.invoke()
        out = interp.get_tensor(out_d['index'])[0].astype(np.float32)
        sc2 = out_d['quantization_parameters']['scales'][0]
        zp2 = out_d['quantization_parameters']['zero_points'][0]
        return (out - zp2) * sc2
    else:
        interp.set_tensor(inp_d['index'], inp)
        interp.invoke()
        return interp.get_tensor(out_d['index'])[0]

def move_to_trash(filepath, reason):
    fname = os.path.basename(filepath)
    dst   = os.path.join(TRASH_DIR, fname)
    if os.path.exists(dst):
        base, ext = os.path.splitext(fname)
        dst = os.path.join(TRASH_DIR, f"{base}_dup{ext}")
    shutil.move(filepath, dst)
    return dst

# ── Hardcoded POLICE blacklist dari audit ────────────────────────────
POLICE_BLACKLIST = {
    "aug_noise_1388_police_0162_loud.wav",
    "aug_noise_1440_police_0385_noise_light.wav",
    "aug_noise_2437_police_0003_original.wav",
    "aug_noise_3004_police_0003_seg01.wav",
    "aug_noise_4135_police_0003_seg14.wav",
    "aug_noise_5200_police_0136_shift.wav",
    "aug_noise_7626_police_0003_seg19.wav",
    "aug_noise_8898_police_0003_shift.wav",
    "aug_noise_9226_police_0403_original.wav",
    "aug_pitch_1638_police_0001_seg56.wav",
    "aug_pitch_6372_police_0062_original.wav",
    "police_0003_seg03.wav",
    "police_0003_seg06.wav",
    "police_0003_seg09.wav",
    "police_0003_seg10.wav",
    "police_0003_seg14.wav",
    "police_0003_seg15.wav",
    "police_0003_seg17.wav",
}

print("=" * 70)
print("  DATASET CLEANING v2 — Membersihkan semua kelas")
print("=" * 70)
print(f"  TRASH dir: {TRASH_DIR}\n")

total_moved = 0
stats = {cat: 0 for cat in CATEGORIES}

# ── Pass 1: Police blacklist (langsung pindahkan tanpa predict) ──────
print("[1/4] Memindahkan file POLICE blacklist (18 file ambigu)...")
for fname in POLICE_BLACKLIST:
    fp = os.path.join(BASE_DIR, "POLICE", fname)
    if os.path.exists(fp):
        move_to_trash(fp, "police_blacklist")
        stats['POLICE'] += 1
        total_moved += 1
print(f"  → Dipindahkan: {stats['POLICE']} file\n")

# ── Pass 2: SYNTH_fire yang diprediksi sebagai AMBULANCE ────────────
print("[2/4] Mengaudit file SYNTH_fire_* di FIRETRUCK...")
synth_fire_files = glob.glob(os.path.join(BASE_DIR, "FIRETRUCK", "SYNTH_fire*.wav"))
moved_synth = 0
for fp in synth_fire_files:
    try:
        y, _ = librosa.load(fp, sr=SAMPLE_RATE)
        probs = predict(extract_features(y))
        pred  = np.argmax(probs)
        # Jika diprediksi AMBULANCE dengan keyakinan tinggi → file sintesis salah
        if pred == 0 and probs[0] > 0.70:
            move_to_trash(fp, "synth_fire_wrong_ambulance")
            moved_synth += 1
            total_moved += 1
    except:
        pass
stats['FIRETRUCK'] = moved_synth
print(f"  → {len(synth_fire_files)} SYNTH_fire diperiksa, {moved_synth} dipindahkan (diprediksi Ambulance)\n")

# ── Pass 3: Scan semua kelas — hapus file yang sangat mislabel ───────
print("[3/4] Scanning semua kelas untuk file mislabel berat...")
print("      (Kriteria: skor kelas sendiri < 15% DAN skor kelas lain > 85%)")

for cat_idx, cat in enumerate(CATEGORIES):
    cat_dir = os.path.join(BASE_DIR, cat)
    all_wav = glob.glob(os.path.join(cat_dir, "*.wav"))
    moved   = 0
    checked = 0
    for fp in all_wav:
        fname = os.path.basename(fp)
        # Skip: file sudah dipindahkan (tidak ada lagi)
        if not os.path.exists(fp):
            continue
        try:
            y, _ = librosa.load(fp, sr=SAMPLE_RATE)
            rms  = np.sqrt(np.mean(y**2))
            # Skip file yang terlalu hening (< 0.0005 RMS) → pasti bad
            if rms < 0.0005:
                move_to_trash(fp, f"too_quiet_rms={rms:.5f}")
                moved += 1
                total_moved += 1
                continue
            probs     = predict(extract_features(y))
            own_score = float(probs[cat_idx])
            pred      = int(np.argmax(probs))
            pred_score= float(probs[pred])
            # Sangat mislabel: skor sendiri < 15% dan kelas lain > 85%
            if own_score < 0.15 and pred_score > 0.85 and pred != cat_idx:
                move_to_trash(fp, f"mislabel_{CATEGORIES[pred]}={pred_score:.2f}_own={own_score:.2f}")
                moved += 1
                total_moved += 1
            checked += 1
        except:
            pass
    stats[cat] += moved
    print(f"  {cat}: {checked} diperiksa, {moved} dipindahkan (mislabel/hening)")

# ── Pass 4: NORMAL files yang diprediksi sirine sangat kuat ─────────
print("\n[4/4] Mengaudit NORMAL files yang diprediksi sebagai sirine >92%...")
normal_files = glob.glob(os.path.join(BASE_DIR, "NORMAL", "*.wav"))
moved_normal = 0
for fp in normal_files:
    if not os.path.exists(fp):
        continue
    try:
        y, _   = librosa.load(fp, sr=SAMPLE_RATE)
        probs  = predict(extract_features(y))
        own    = float(probs[2])  # NORMAL class
        pred   = int(np.argmax(probs))
        pred_s = float(probs[pred])
        # Jika diprediksi sirine dengan keyakinan > 92% → kemungkinan besar sirine nyata yang salah folder
        if pred != 2 and pred_s > 0.92:
            move_to_trash(fp, f"normal_predicted_as_{CATEGORIES[pred]}={pred_s:.2f}")
            moved_normal += 1
            total_moved += 1
    except:
        pass
stats['NORMAL'] += moved_normal
print(f"  → {moved_normal} NORMAL files dipindahkan (kemungkinan salah folder)\n")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  HASIL PEMBERSIHAN DATASET")
print("=" * 70)
for cat in CATEGORIES:
    remaining = len(glob.glob(os.path.join(BASE_DIR, cat, "*.wav")))
    print(f"  {cat:10s}: {stats[cat]:4d} file dihapus, {remaining:5d} tersisa")
print(f"\n  TOTAL dipindahkan ke TRASH: {total_moved} file")
print(f"  Lokasi TRASH: {TRASH_DIR}")
print("\n  Dataset siap untuk di-retrain!")
print("  Jalankan: python 04_Training_AI/train_v2_fix_police.py")
