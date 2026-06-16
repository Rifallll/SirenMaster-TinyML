import sounddevice as sd
import numpy as np
import time

print("Testing sounddevice playrec...")
# Generate 1 second of 440Hz sine wave
sr = 8000
t = np.linspace(0, 1.0, sr, endpoint=False)
y = 0.5 * np.sin(2 * np.pi * 440 * t)

print("Playing and recording for 1 second...")
try:
    rec = sd.playrec(y, samplerate=sr, channels=1)
    sd.wait()
    print("Success! Recorded shape:", rec.shape)
    print("Recorded max amplitude:", np.max(np.abs(rec)))
except Exception as e:
    print("Error during playrec:", e)
