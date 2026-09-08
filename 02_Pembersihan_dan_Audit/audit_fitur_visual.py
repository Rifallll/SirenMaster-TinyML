"""
audit_fitur_visual.py
=====================
Visualisasi mel-spectrogram dan overlap antar kelas:
- Tampilkan contoh spectrogram tiap kelas
- Tampilkan rata-rata energi tiap kelas per frekuensi
- Scatter plot t-SNE untuk melihat separasi kelas di feature space
"""

import numpy as np, librosa, os, sys, glob, re, json, warnings
import matplotlib
matplotlib.use('Agg')  # headless mode
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Config ──────────────────────────────────────────────────────────
BASE_DIR    = r"C:\Users\ASUS\Videos\DATASET"
MODEL_H     = os.path.join(BASE_DIR, "sirenmaster_main", "model.h")
OUT_DIR     = os.path.join(BASE_DIR, "audit_visual_output")
SAMPLE_RATE = 8000
DURATION    = 4.0
TARGET_LEN  = int(SAMPLE_RATE * DURATION)
CATEGORIES  = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
COLORS      = ['#e74c3c', '#f39c12', '#27ae60', '#2980b9']
N_SAMPLES   = 20  # samples per class for stats
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load scaler ──────────────────────────────────────────────────────
with open(MODEL_H, 'r') as f:
    content = f.read()
mean_m = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\}', content)
std_m  = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\}',  content)
MEL_MEAN = np.array([float(x.strip().rstrip('f')) for x in mean_m.group(1).split(',')])
MEL_STD  = np.array([float(x.strip().rstrip('f')) for x in std_m.group(1).split(',')])

def extract_raw_melspec(y):
    """Extract mel-spectrogram (RAW, before z-score normalization) for visualization."""
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
    return np.array(frames, dtype=np.float32)  # (249, 40)

# ──────────────────────────────────────────────────────────────────────
# 1. GAMBAR CONTOH SPECTROGRAM per kelas (3x4 grid)
# ──────────────────────────────────────────────────────────────────────
print("[1/4] Membuat contoh spectrogram per kelas...")
fig, axes = plt.subplots(4, 3, figsize=(18, 14))
fig.suptitle('Contoh Mel-Spectrogram per Kelas Sirine\n(Semakin terang = Energi lebih tinggi)', 
             fontsize=15, fontweight='bold', y=1.01)

for cat_idx, cat in enumerate(CATEGORIES):
    files = sorted(glob.glob(os.path.join(BASE_DIR, cat, "*.wav")))
    # Skip synthetic files
    clean = [f for f in files if not os.path.basename(f).startswith('SYNTH_')]
    selected = clean[:3] if len(clean) >= 3 else clean

    for col, fp in enumerate(selected):
        ax = axes[cat_idx][col]
        try:
            y, _ = librosa.load(fp, sr=SAMPLE_RATE)
            spec  = extract_raw_melspec(y)  # (249, 40)
            im = ax.imshow(spec.T, aspect='auto', origin='lower',
                           cmap='inferno', interpolation='nearest',
                           vmin=-15, vmax=5)
            ax.set_title(f"{cat}\n{os.path.basename(fp)[:30]}", fontsize=8)
            ax.set_ylabel("Mel Band (Hz)" if col == 0 else "")
            ax.set_xlabel("Time Frame")
            # Add mel freq labels
            mel_freqs = librosa.mel_frequencies(n_mels=40, fmin=0, fmax=4000)
            ax.set_yticks([0, 10, 20, 30, 39])
            ax.set_yticklabels([f"{mel_freqs[i]:.0f}" for i in [0, 10, 20, 30, 39]], fontsize=7)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:40]}', transform=ax.transAxes, ha='center', fontsize=7)
            ax.set_title(f"{cat} - ERROR", fontsize=8)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "1_spectrogram_contoh.png")
plt.savefig(out1, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {out1}")

# ──────────────────────────────────────────────────────────────────────
# 2. RATA-RATA ENERGI MEL per KELAS (overlay — menunjukkan OVERLAP)
# ──────────────────────────────────────────────────────────────────────
print("[2/4] Menghitung rata-rata profil frekuensi per kelas...")
mel_freqs = librosa.mel_frequencies(n_mels=40, fmin=0, fmax=4000)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Profil Frekuensi Rata-rata per Kelas\n(Puncak yang berdekatan = Mudah terkonfusikan)',
             fontsize=14, fontweight='bold')

ax1 = axes[0]  # overlay all classes
ax2 = axes[1]  # std range shading

class_profiles = {}
class_stds = {}

for cat_idx, cat in enumerate(CATEGORIES):
    files = sorted(glob.glob(os.path.join(BASE_DIR, cat, "*.wav")))
    clean = [f for f in files if not os.path.basename(f).startswith('SYNTH_')]
    selected = clean[:N_SAMPLES]
    
    all_means = []
    for fp in selected:
        try:
            y, _ = librosa.load(fp, sr=SAMPLE_RATE)
            spec  = extract_raw_melspec(y)  # (249, 40)
            # Average over time → (40,)
            mean_freq = np.mean(spec, axis=0)
            all_means.append(mean_freq)
        except:
            pass
    
    if all_means:
        stack = np.array(all_means)
        profile = np.mean(stack, axis=0)
        std_dev = np.std(stack, axis=0)
        class_profiles[cat] = profile
        class_stds[cat] = std_dev
        
        color = COLORS[cat_idx]
        # Plot 1: overlay mean
        ax1.plot(mel_freqs, profile, color=color, linewidth=2.5, label=cat)
        # Plot 2: mean + std shading
        ax2.plot(mel_freqs, profile, color=color, linewidth=2.5, label=cat)
        ax2.fill_between(mel_freqs, profile - std_dev, profile + std_dev,
                         alpha=0.2, color=color)

ax1.set_xlabel("Frekuensi Mel (Hz)", fontsize=12)
ax1.set_ylabel("Log-Energi Rata-rata", fontsize=12)
ax1.set_title("Overlay Profil Frekuensi\n(Garis dekat = Overlap tinggi = Sering tertukar!)", fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 4000])
ax1.axvspan(500, 1500, alpha=0.08, color='red', label='Zona Overlap Sirine')
ax1.text(850, ax1.get_ylim()[0], 'ZONA\nOVERLAP', fontsize=9, color='red', ha='center')

ax2.set_xlabel("Frekuensi Mel (Hz)", fontsize=12)
ax2.set_ylabel("Log-Energi ± Std Dev", fontsize=12)
ax2.set_title("Rentang Variasi per Kelas\n(Bayangan lebar = Variasi tinggi = Sulit dibedakan)", fontsize=11)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 4000])

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "2_profil_frekuensi.png")
plt.savefig(out2, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {out2}")

# ──────────────────────────────────────────────────────────────────────
# 3. COSINE SIMILARITY MATRIX antar kelas (menunjukkan seberapa mirip)
# ──────────────────────────────────────────────────────────────────────
print("[3/4] Menghitung similarity matrix antar kelas...")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

if len(class_profiles) == 4:
    sim_matrix = np.zeros((4, 4))
    for i, cat_i in enumerate(CATEGORIES):
        for j, cat_j in enumerate(CATEGORIES):
            if cat_i in class_profiles and cat_j in class_profiles:
                sim_matrix[i][j] = cosine_similarity(class_profiles[cat_i], class_profiles[cat_j])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0.9, vmax=1.0)
    ax.set_xticks(range(4)); ax.set_xticklabels(CATEGORIES, fontsize=12)
    ax.set_yticks(range(4)); ax.set_yticklabels(CATEGORIES, fontsize=12)
    ax.set_title('Similarity Matrix Antar Kelas\n(Mendekati 1.0 = Sangat mirip = Mudah tertukar!)', 
                 fontsize=13, fontweight='bold')

    for i in range(4):
        for j in range(4):
            val  = sim_matrix[i][j]
            text = f"{val:.4f}"
            color = 'white' if val > 0.97 else 'black'
            level = ""
            if i != j:
                if val > 0.99: level = "\n⚠️ SANGAT MIRIP!"
                elif val > 0.97: level = "\n⚡ Mirip"
            ax.text(j, i, text + level, ha='center', va='center', fontsize=10, 
                    color=color, fontweight='bold' if val > 0.97 else 'normal')
    
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    plt.tight_layout()
    out3 = os.path.join(OUT_DIR, "3_similarity_matrix.png")
    plt.savefig(out3, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out3}")

    # Print similarity findings
    print("\n  === TEMUAN OVERLAP ANTAR KELAS ===")
    for i in range(4):
        for j in range(i+1, 4):
            sim = sim_matrix[i][j]
            tag = "SANGAT MIRIP ⚠️" if sim > 0.99 else ("Mirip ⚡" if sim > 0.97 else "OK")
            print(f"  {CATEGORIES[i]:10s} <-> {CATEGORIES[j]:10s}: {sim:.4f}  {tag}")

# ──────────────────────────────────────────────────────────────────────
# 4. WAVEFORM COMPARISON: Fire vs Ambulance (kenapa tertukar)
# ──────────────────────────────────────────────────────────────────────
print("\n[4/4] Membandingkan waveform Fire vs Ambulance...")
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
fig.suptitle('Perbandingan Audio: FIRETRUCK vs AMBULANCE\n(Kenapa mudah tertukar?)',
             fontsize=14, fontweight='bold')

for row, (cat, cat_idx) in enumerate([('FIRETRUCK', 1), ('AMBULANCE', 0)]):
    files = sorted(glob.glob(os.path.join(BASE_DIR, cat, "*.wav")))
    clean = [f for f in files if not f.startswith('SYNTH_')][:1]
    
    if not clean:
        continue
    fp = clean[0]
    try:
        y, _ = librosa.load(fp, sr=SAMPLE_RATE)
        y4   = y[:TARGET_LEN] if len(y) >= TARGET_LEN else np.pad(y, (0, TARGET_LEN - len(y)))
        
        # Col 0: waveform
        ax = axes[row][0]
        t  = np.linspace(0, DURATION, len(y4))
        ax.plot(t, y4, color=COLORS[cat_idx], linewidth=0.5, alpha=0.8)
        ax.set_title(f"{cat} — Waveform", fontsize=11)
        ax.set_xlabel("Waktu (s)")
        ax.set_ylabel("Amplitudo")
        ax.set_xlim([0, DURATION])
        ax.grid(True, alpha=0.3)
        
        # Col 1: mel-spectrogram
        spec = extract_raw_melspec(y4)
        ax = axes[row][1]
        ax.imshow(spec.T, aspect='auto', origin='lower', cmap='inferno',
                  interpolation='nearest', vmin=-15, vmax=5)
        ax.set_title(f"{cat} — Mel-Spectrogram", fontsize=11)
        ax.set_xlabel("Frame (waktu)")
        ax.set_ylabel("Mel Band")
        
        # Col 2: rata-rata spektrum frekuensi
        ax = axes[row][2]
        mean_spec = np.mean(spec, axis=0)
        ax.fill_between(mel_freqs, mean_spec, alpha=0.4, color=COLORS[cat_idx])
        ax.plot(mel_freqs, mean_spec, color=COLORS[cat_idx], linewidth=2, label=cat)
        ax.set_title(f"{cat} — Profil Frekuensi", fontsize=11)
        ax.set_xlabel("Frekuensi (Hz)")
        ax.set_ylabel("Log-Energi")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 4000])
        
    except Exception as e:
        print(f"  Error processing {cat}: {e}")

plt.tight_layout()
out4 = os.path.join(OUT_DIR, "4_fire_vs_ambulance.png")
plt.savefig(out4, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {out4}")

print("\n" + "=" * 60)
print("  SELESAI! Output di:")
print(f"  {OUT_DIR}")
print("  File yang dibuat:")
print("    1_spectrogram_contoh.png  — Contoh mel-spec tiap kelas")
print("    2_profil_frekuensi.png    — Overlay frekuensi + overlap zone")
print("    3_similarity_matrix.png   — Seberapa mirip tiap kelas")
print("    4_fire_vs_ambulance.png   — Perbandingan FIRE vs AMBULANCE")
print("=" * 60)
