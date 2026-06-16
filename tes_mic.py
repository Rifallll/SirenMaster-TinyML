import os
import sys
import time
import numpy as np

try:
    import sounddevice as sd
    import scipy.io.wavfile as wav
except ImportError:
    print("ERROR: pip install sounddevice scipy")
    sys.exit(1)

SAMPLE_RATE = 8000
DURATION = 5.0

print("=" * 60)
print("  DIAGNOSTIK MICROPHONE LAPTOP")
print("=" * 60)
print("Persiapan: Dekatkan HP Anda yang memutar suara sirine ke mic laptop.")
print("Script ini akan merekam selama 5 detik...")
print("Mulai perekaman dalam:")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

print("\n🔴 MEREKAM... (Silakan putar suara sirine sekarang!)")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
print("✅ Selesai Merekam.")

max_amp = np.max(np.abs(audio))
rms = np.sqrt(np.mean(audio**2))

print("\n" + "=" * 60)
print("  HASIL ANALISIS AUDIO:")
print("=" * 60)
print(f"  Maksimum Amplitudo : {max_amp:.6f}")
print(f"  RMS Volume Level   : {rms:.6f}")

if max_amp < 0.001:
    print("\n❌ SANGAT HENING: Microphone tidak menerima suara atau mute!")
    print("   Periksa pengaturan microphone Windows Anda (pastikan tidak di-mute).")
elif max_amp < 0.01:
    print("\n⚠️ TERLALU PELAN: Suara terdeteksi tapi sangat lemah.")
    print("   Dekatkan HP ke microphone laptop atau naikkan volume HP Anda.")
else:
    print("\n✅ SUARA CUKUP KERAS: Microphone berfungsi dengan baik!")
    
# Save to file for debug
filename = "debug_mic_test.wav"
wav_data = np.int16(audio / (max_amp + 1e-9) * 32767)
wav.write(filename, SAMPLE_RATE, wav_data)
print(f"  Audio tes disimpan ke: {filename}")
print("=" * 60)
