import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

dataset_dir = r"C:\Users\ASUS\Videos\DATASET"
classes = ["AMBULANCE", "FIRETRUCK", "POLICE", "NORMAL"]
artifact_dir = r"C:\Users\ASUS\.gemini\antigravity\brain\36105c52-8459-4796-a547-7c54aea5e73e"

plt.figure(figsize=(16, 10))

for i, cls in enumerate(classes):
    cls_dir = os.path.join(dataset_dir, cls)
    if not os.path.exists(cls_dir):
        continue
    files = [f for f in os.listdir(cls_dir) if f.endswith('.wav')]
    if not files:
        continue
    sample_file = os.path.join(cls_dir, random.choice(files))
    
    # Load audio 4 detik
    y, sr = librosa.load(sample_file, sr=8000, duration=4.0)
    
    # Extract Mel Spectrogram (sama persis dengan setingan AI dan C++ kita)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=256, hop_length=128, n_mels=40, fmin=300, fmax=3500)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.subplot(2, 2, i+1)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmin=300, fmax=3500, hop_length=128, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Karakteristik Visual Suara: {cls}')
    plt.xlabel('Waktu (detik)')
    plt.ylabel('Frekuensi Mel (Hz)')

plt.tight_layout()
output_path = os.path.join(artifact_dir, "spectrogram_comparison.png")
plt.savefig(output_path, dpi=150)
print(f"Berhasil menyimpan plot ke {output_path}")
