import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os
import glob
import numpy as np
import scipy.io.wavfile as wavfile

ROOT = r"C:\Users\ASUS\Videos\DATASET"
TARGET_FOLDERS = ["AMBULANCE", "POLICE", "FIRETRUCK"]

print("=" * 70)
print("🔊 MAKSIMALISASI VOLUME DATASET SIRINE (TANPA FILTER HPF)")
print("   Hanya Peak Normalization — agar cocok dengan ESP32 inferensi")
print("=" * 70)

total_boosted = 0
total_processed = 0

for folder in TARGET_FOLDERS:
    folder_path = os.path.join(ROOT, folder)
    wav_files = sorted(glob.glob(os.path.join(folder_path, "*.wav")))
    print(f"\n[*] Memproses folder {folder} ({len(wav_files)} file)...")

    boosted_count = 0

    for fpath in wav_files:
        total_processed += 1
        try:
            sr, data = wavfile.read(fpath)
        except Exception as e:
            continue

        if len(data.shape) > 1:
            data = data[:, 0]

        data_float = data.astype(np.float32)

        # Peak Normalization ke 95% (-0.45 dB) TANPA HPF
        peak = np.max(np.abs(data_float))
        if peak > 10.0:
            target_peak = 31128.0  # 95% dari 32767
            gain = target_peak / peak
            gain = min(gain, 20.0)  # Batasi max gain 20x
            data_boosted = data_float * gain
            data_int16 = np.clip(data_boosted, -32767, 32767).astype(np.int16)
            wavfile.write(fpath, sr, data_int16)
            boosted_count += 1
            total_boosted += 1

    print(f"  ✅ Selesai: {boosted_count} file di {folder} berhasil dinormalisasi.")

print("\n" + "=" * 70)
print(f"🎉 TOTAL: {total_boosted} dari {total_processed} file berhasil dinormalisasi!")
print("   Volume sirine maksimal, spektrum asli tetap utuh (tanpa HPF).")
print("=" * 70)
