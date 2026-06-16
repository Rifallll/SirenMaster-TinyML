import os
import time
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import tensorflow as tf
import sys

SAMPLE_RATE = 8000
RECORD_SR = 44100  # Record at native 44100Hz to avoid bad PortAudio resampling
DURATION = 4.0
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
MODEL_PATH = "siren_model_quant.tflite"
CACHE_PATH = "siren_40x249_melspec_cache_lokal.npz"

print("Recording 4 seconds at 44100Hz... Please play the siren sound now!")
time.sleep(1.0)
print("RECORDING STARTED...")
audio_data_44k = sd.rec(int(RECORD_SR * DURATION), samplerate=RECORD_SR, channels=1, dtype='float32')
sd.wait()
print("RECORDING FINISHED.")

audio_data_44k = audio_data_44k[:, 0]

# Resample to 8000Hz using librosa (high quality)
print("Resampling from 44100Hz to 8000Hz in software...")
audio_data = librosa.resample(audio_data_44k, orig_sr=RECORD_SR, target_sr=SAMPLE_RATE)

# Save to scratch directory
os.makedirs("scratch", exist_ok=True)
record_path = "scratch/recorded_resampled.wav"
wav.write(record_path, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
print(f"Saved resampled recording to {record_path}")

# Check signal properties
max_val = np.max(np.abs(audio_data))
mean_val = np.mean(audio_data)
rms_val = np.sqrt(np.mean(audio_data**2))
print(f"Signal Properties:")
print(f"  Max Amplitude: {max_val:.6f}")
print(f"  Mean (DC):     {mean_val:.6f}")
print(f"  RMS:           {rms_val:.6f}")

sys.path.append(os.path.dirname(__file__))
from debug_test import predict_audio_file
probs = predict_audio_file(record_path)
pred = CATEGORIES[np.argmax(probs)]
print(f"\nPrediction: {pred}")
print("Probabilities:")
for i, cat in enumerate(CATEGORIES):
    print(f"  {cat}: {probs[i]*100:.2f}%")
