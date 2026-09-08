"""
visualisasi_spektrogram_audit.py
=================================
Buat gambar spektrogram dari file-file yang salah kamar
untuk diperiksa secara visual.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT    = r"C:\Users\ASUS\Videos\DATASET"
OUT_DIR = os.path.join(ROOT, "02_Pembersihan_dan_Audit", "spektrogram_audit")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_RATE = 8000
N_FFT       = 256
HOP_LEN     = 128
N_MELS      = 40

# ── File sampel yang ditemukan audit (salah kamar) ────────────────────────
SALAH_KAMAR = {
    # AMBULANCE files yang diprediksi FIRETRUCK / POLICE
    "AMB→FIRE_AMB_0085": ("AMBULANCE", "ambulance_0085_seg01.wav"),
    "AMB→FIRE_AMB_0110": ("AMBULANCE", "ambulance_0110_seg01.wav"),
    "AMB→POL_AMB_0058":  ("AMBULANCE", "ambulance_0058_seg01.wav"),
    "AMB→POL_AMB_0094":  ("AMBULANCE", "ambulance_0094_seg01.wav"),

    # POLICE files yang diprediksi AMBULANCE
    "POL→AMB_SYNTH_6":   ("POLICE",    "SYNTH_pol_6.wav"),
    "POL→AMB_SYNTH_14":  ("POLICE",    "SYNTH_pol_14.wav"),
    "POL→AMB_aug_1054":  ("POLICE",    "aug_noise_1054_police_0297_seg01.wav"),

    # FIRETRUCK files yang diprediksi AMBULANCE
    "FIRE→AMB_dl_0003":  ("FIRETRUCK", "dl_v3_dam_new_01_part0003.wav"),
    "FIRE→AMB_dl_0040":  ("FIRETRUCK", "dl_v3_dam_new_01_part0040.wav"),
    "FIRE→AMB_dl_0066":  ("FIRETRUCK", "dl_v3_dam_new_01_part0066.wav"),
}

# File referensi yang BENAR (diambil dari setiap kelas)
REFERENSI = {
    "BENAR_AMB":  ("AMBULANCE", "ambulance_0001_seg01.wav"),
    "BENAR_POL":  ("POLICE",    "police_0001_seg01.wav"),
    "BENAR_FIRE": ("FIRETRUCK", "firetruck_0001_seg01.wav"),
}

def get_any_wav(folder):
    """Ambil file wav pertama dari folder."""
    import glob
    files = sorted(glob.glob(os.path.join(ROOT, folder, "*.wav")))
    return files[0] if files else None

def load_mel(fpath):
    audio, _ = librosa.load(fpath, sr=SAMPLE_RATE, mono=True, duration=3.0)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max)

def plot_satu(ax, fpath, title, color):
    try:
        mel_db = load_mel(fpath)
        librosa.display.specshow(mel_db, sr=SAMPLE_RATE, hop_length=HOP_LEN,
                                 x_axis='time', y_axis='mel', ax=ax, cmap='magma')
        ax.set_title(title, fontsize=8, color=color, fontweight='bold')
        ax.set_xlabel('Waktu (s)', fontsize=7)
        ax.set_ylabel('Mel', fontsize=7)
        ax.tick_params(labelsize=6)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center',
                transform=ax.transAxes, fontsize=7)
        ax.set_title(title, fontsize=8)

# ── Buat gambar 1: Referensi BENAR per kelas ──────────────────────────────
print("Membuat gambar referensi kelas yang BENAR...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("REFERENSI: Spektrogram Sirine yang BENAR per Kelas", fontsize=12, fontweight='bold')
colors = {'AMBULANCE': '#00BFFF', 'POLICE': '#FFD700', 'FIRETRUCK': '#FF4500'}

for idx, (key, (folder, fname)) in enumerate(REFERENSI.items()):
    fpath = os.path.join(ROOT, folder, fname)
    if not os.path.exists(fpath):
        fpath = get_any_wav(folder)
    if fpath:
        plot_satu(axes[idx], fpath, f"✅ {folder}\n{os.path.basename(fpath)}", colors[folder])

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "01_referensi_benar.png")
plt.savefig(out1, dpi=130, bbox_inches='tight')
plt.close()
print(f"  Tersimpan: {out1}")

# ── Buat gambar 2: File AMBULANCE yang salah → POLICE/FIRE ────────────────
print("Membuat gambar AMBULANCE yang salah kamar...")
salah_amb = [(k,v) for k,v in SALAH_KAMAR.items() if v[0]=="AMBULANCE"]
n = len(salah_amb)
fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
if n == 1: axes = [axes]
fig.suptitle("❌ File di folder AMBULANCE tapi diprediksi kelas LAIN", fontsize=11, fontweight='bold', color='#FF4444')

for idx, (key, (folder, fname)) in enumerate(salah_amb):
    fpath = os.path.join(ROOT, folder, fname)
    pred_kelas = key.split("→")[1].split("_")[0]
    title = f"📁 {folder}\n→ model: {pred_kelas}\n{fname[:30]}"
    plot_satu(axes[idx], fpath, title, '#FF6B6B')

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "02_ambulance_salah.png")
plt.savefig(out2, dpi=130, bbox_inches='tight')
plt.close()
print(f"  Tersimpan: {out2}")

# ── Buat gambar 3: File POLICE yang salah → AMBULANCE ────────────────────
print("Membuat gambar POLICE yang salah kamar...")
salah_pol = [(k,v) for k,v in SALAH_KAMAR.items() if v[0]=="POLICE"]
n = len(salah_pol)
fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
if n == 1: axes = [axes]
fig.suptitle("❌ File di folder POLICE tapi diprediksi AMBULANCE", fontsize=11, fontweight='bold', color='#FFA500')

for idx, (key, (folder, fname)) in enumerate(salah_pol):
    fpath = os.path.join(ROOT, folder, fname)
    title = f"📁 {folder}\n→ model: AMBULANCE\n{fname[:30]}"
    plot_satu(axes[idx], fpath, title, '#FFA500')

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "03_police_salah.png")
plt.savefig(out3, dpi=130, bbox_inches='tight')
plt.close()
print(f"  Tersimpan: {out3}")

# ── Buat gambar 4: File FIRETRUCK yang salah → AMBULANCE ─────────────────
print("Membuat gambar FIRETRUCK yang salah kamar...")
salah_fire = [(k,v) for k,v in SALAH_KAMAR.items() if v[0]=="FIRETRUCK"]
n = len(salah_fire)
fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
if n == 1: axes = [axes]
fig.suptitle("❌ File di folder FIRETRUCK tapi diprediksi AMBULANCE", fontsize=11, fontweight='bold', color='#FF4500')

for idx, (key, (folder, fname)) in enumerate(salah_fire):
    fpath = os.path.join(ROOT, folder, fname)
    title = f"📁 {folder}\n→ model: AMBULANCE\n{fname[:30]}"
    plot_satu(axes[idx], fpath, title, '#FF7043')

plt.tight_layout()
out4 = os.path.join(OUT_DIR, "04_firetruck_salah.png")
plt.savefig(out4, dpi=130, bbox_inches='tight')
plt.close()
print(f"  Tersimpan: {out4}")

# ── Buat gambar 5: Perbandingan side-by-side BENAR vs SALAH ──────────────
print("Membuat gambar perbandingan besar...")
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.35)
fig.suptitle("PERBANDINGAN SPEKTROGRAM: BENAR vs SALAH KAMAR\n(Lihat pola berbeda antar kelas)", 
             fontsize=13, fontweight='bold')

# Baris 1: AMBULANCE benar + 3 yang salah
ax = fig.add_subplot(gs[0, 0])
fp = get_any_wav("AMBULANCE")
plot_satu(ax, fp, "✅ AMBULANCE BENAR", '#00BFFF')

for col, (key, (folder, fname)) in enumerate(salah_amb[:3], start=1):
    ax = fig.add_subplot(gs[0, col])
    fpath = os.path.join(ROOT, folder, fname)
    pred_kelas = key.split("→")[1].split("_")[0]
    plot_satu(ax, fpath, f"❌ AMB tapi→{pred_kelas}\n{fname[:25]}", '#FF6B6B')

# Baris 2: POLICE benar + 3 yang salah
ax = fig.add_subplot(gs[1, 0])
fp = get_any_wav("POLICE")
plot_satu(ax, fp, "✅ POLICE BENAR", '#FFD700')

for col, (key, (folder, fname)) in enumerate(salah_pol[:3], start=1):
    ax = fig.add_subplot(gs[1, col])
    fpath = os.path.join(ROOT, folder, fname)
    plot_satu(ax, fpath, f"❌ POL tapi→AMB\n{fname[:25]}", '#FFA500')

# Baris 3: FIRETRUCK benar + 3 yang salah
ax = fig.add_subplot(gs[2, 0])
fp = get_any_wav("FIRETRUCK")
plot_satu(ax, fp, "✅ FIRETRUCK BENAR", '#FF4500')

for col, (key, (folder, fname)) in enumerate(salah_fire[:3], start=1):
    ax = fig.add_subplot(gs[2, col])
    fpath = os.path.join(ROOT, folder, fname)
    plot_satu(ax, fpath, f"❌ FIRE tapi→AMB\n{fname[:25]}", '#FF7043')

out5 = os.path.join(OUT_DIR, "05_perbandingan_lengkap.png")
plt.savefig(out5, dpi=130, bbox_inches='tight')
plt.close()
print(f"  Tersimpan: {out5}")

print("\n" + "="*60)
print(f"✅ Selesai! Semua gambar disimpan di:")
print(f"   {OUT_DIR}")
print("\nFile yang dibuat:")
for f in ["01_referensi_benar.png", "02_ambulance_salah.png",
          "03_police_salah.png", "04_firetruck_salah.png", "05_perbandingan_lengkap.png"]:
    print(f"   - {f}")
