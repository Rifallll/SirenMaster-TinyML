"""
cek_frekuensi_puncak.py
=======================
Menganalisis frekuensi puncak (peak frequency bin) dari semua file sirine 
dalam dataset untuk memvalidasi perbedaan rentang frekuensi antara vokal manusia 
vs sirine nyata.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, glob, numpy as np, librosa

ROOT = r"C:\Users\ASUS\Videos\DATASET"

siren_classes = ['AMBULANCE', 'FIRETRUCK', 'POLICE']
all_peaks = []

print("=" * 80)
print(" ANALISIS FREKUENSI PUNCAK (PEAK FREQUENCY) SIRINE")
print("=" * 80)
print(f"{'Kelas':<12} | {'File':<40} | {'Peak Freq (Hz)':<18}")
print("-" * 80)

for cls in siren_classes:
    files = sorted(glob.glob(os.path.join(ROOT, cls, "*.wav")))
    for fp in files[:10]: # Cek 10 file pertama tiap kelas
        fn = os.path.basename(fp)
        y, sr = librosa.load(fp, sr=8000)
        y -= np.mean(y)
        
        # Hitung FFT rata-rata sepanjang file
        # Bagi menjadi 32 frame untuk kestabilan
        frame_len = 256
        hop_len = 128
        n_frames = (len(y) - frame_len) // hop_len + 1
        
        peaks = []
        for f in range(n_frames):
            frame = y[f*hop_len : f*hop_len + frame_len]
            if len(frame) < frame_len: break
            fft_vals = np.abs(np.fft.rfft(frame))**2
            # Cari bin tertinggi (abaikan DC / sub-bass di bawah 200Hz -> bin 0 sampai 6)
            peak_bin = np.argmax(fft_vals[6:]) + 6
            freq = peak_bin * (8000 / 256)
            peaks.append(freq)
        
        avg_peak = np.mean(peaks) if peaks else 0
        all_peaks.append(avg_peak)
        print(f"{cls:<12} | {fn:<40} | {avg_peak:.1f} Hz")

print("-" * 80)
print(f"Rata-rata frekuensi puncak semua sirine: {np.mean(all_peaks):.1f} Hz")
print(f"Batas bawah frekuensi puncak terkecil:  {np.min(all_peaks):.1f} Hz")
print(f"Batas atas frekuensi puncak terbesar:  {np.max(all_peaks):.1f} Hz")
print("=" * 80)
