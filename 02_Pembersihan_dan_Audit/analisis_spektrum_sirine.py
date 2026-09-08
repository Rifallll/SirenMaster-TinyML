"""
analisis_spektrum_sirine.py
===========================
Bandingkan distribusi energi spektral frekuensi tinggi antara sirine nyata 
vs suara manusia (mimikri/vokal).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, glob, numpy as np, librosa

ROOT = r"C:\Users\ASUS\Videos\DATASET"

# Cari file sirine
files = {
    'AMBULANCE': sorted(glob.glob(os.path.join(ROOT, "AMBULANCE", "*.wav")))[:5],
    'FIRETRUCK': sorted(glob.glob(os.path.join(ROOT, "FIRETRUCK", "*.wav")))[:5],
    'POLICE': sorted(glob.glob(os.path.join(ROOT, "POLICE", "*.wav")))[:5],
}

print("=" * 85)
print(" ANALISIS ENERGI SPEKTRAL FREKUENSI TINGGI (1.5 kHz - 4.0 kHz)")
print("=" * 85)
print(f"{'Kelas':<12} | {'File':<30} | {'HF Energy %':<15} | {'LF Energy %':<15} | Ratio HF/LF")
print("-" * 85)

for class_name, paths in files.items():
    for fp in paths:
        fn = os.path.basename(fp)
        y, sr = librosa.load(fp, sr=8000)
        y -= np.mean(y)
        
        # Hitung FFT
        fft_vals = np.abs(np.fft.rfft(y, n=256))**2
        
        # Bins: 256 FFT -> rfft has 129 bins (0 Hz to 4000 Hz)
        # Low Frequency (LF): 100 Hz to 1200 Hz -> bins 3 to 38
        # High Frequency (HF): 1500 Hz to 3800 Hz -> bins 48 to 121
        lf_energy = np.sum(fft_vals[3:38])
        hf_energy = np.sum(fft_vals[48:121])
        total_energy = np.sum(fft_vals[3:121]) + 1e-9
        
        ratio = hf_energy / (lf_energy + 1e-9)
        hf_pct = hf_energy / total_energy * 100
        lf_pct = lf_energy / total_energy * 100
        
        print(f"{class_name:<12} | {fn:<30} | {hf_pct:.1f}%          | {lf_pct:.1f}%          | {ratio:.4f}")

print("=" * 85)
