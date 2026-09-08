"""
analisis_kontinuitas_sirene.py
==============================
Menganalisis keragaman amplitudo (kontinuitas) dari sirene vs percakapan/ngobrol/suara acak.
Menggunakan pembagian audio 4 detik menjadi 32 blok (masing-masing 1000 sampel).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, glob, numpy as np, librosa

ROOT = r"C:\Users\ASUS\Videos\DATASET"

def hitung_kontinuitas(y, block_size=1000):
    n_blocks = len(y) // block_size
    rms_vals = []
    for i in range(n_blocks):
        block = y[i*block_size:(i+1)*block_size]
        rms = np.sqrt(np.mean(block**2))
        rms_vals.append(rms)
    rms_vals = np.array(rms_vals)
    max_rms = np.max(rms_vals)
    min_rms = np.min(rms_vals)
    avg_rms = np.mean(rms_vals)
    ratio = min_rms / (max_rms + 1e-9)
    # Standar deviasi relatif
    rel_std = np.std(rms_vals) / (avg_rms + 1e-9)
    return ratio, rel_std, min_rms, max_rms

# Cari file sirene
siren_files = {
    'AMBULANCE': sorted(glob.glob(os.path.join(ROOT, "AMBULANCE", "*.wav")))[:5],
    'FIRETRUCK': sorted(glob.glob(os.path.join(ROOT, "FIRETRUCK", "*.wav")))[:5],
    'POLICE': sorted(glob.glob(os.path.join(ROOT, "POLICE", "*.wav")))[:5],
}

# Simulasi suara ngobrol/speech (kita buat simulasi speech dengan memotong-motong audio atau memotong jeda hening)
print("=" * 80)
print(" ANALISIS KONTINUITAS AMPLITUDO SIRENE")
print("=" * 80)
print(f"{'Kelas':<12} | {'File':<30} | {'Min/Max Ratio':<15} | {'Rel Std (RSD)':<15} | Status")
print("-" * 80)

for class_name, files in siren_files.items():
    for fp in files:
        fn = os.path.basename(fp)
        y, _ = librosa.load(fp, sr=8000)
        # Ambil 4 detik pertama (32000 sampel)
        y = y[:32000]
        if len(y) < 32000:
            y = np.pad(y, (0, 32000 - len(y)))
        ratio, rel_std, _, _ = hitung_kontinuitas(y)
        print(f"{class_name:<12} | {fn:<30} | {ratio:.4f}          | {rel_std:.4f}          | SIRENE")

# Simulasi suara percakapan (ngobrol): amplitudo naik turun tajam dengan jeda sunyi
np.random.seed(42)
print("-" * 80)
print(" SIMULASI SUARA NGOBROL / SPEECH (Jeda suku kata/bicara):")
print("-" * 80)
for i in range(5):
    # Buat sinyal speech buatan: blok suara diselingi blok sunyi
    y_speech = np.zeros(32000)
    for b in range(32):
        if np.random.rand() > 0.4:  # 60% blok ada suara ngobrol
            # Frekuensi vokal acak
            t = np.arange(1000) / 8000
            y_speech[b*1000:(b+1)*1000] = np.sin(2 * np.pi * np.random.randint(150, 800) * t) * np.random.rand()
    ratio, rel_std, _, _ = hitung_kontinuitas(y_speech)
    print(f"{'NGOBROL':<12} | {f'simulasi_ngobrol_{i}':<30} | {ratio:.4f}          | {rel_std:.4f}          | SPEECH")

print("=" * 80)
