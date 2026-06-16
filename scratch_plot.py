import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Set parameters matching the DSP config
SAMPLE_RATE = 8000
N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40
DURATION = 1.024

def extract_melspec(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    target_len = int(SAMPLE_RATE * DURATION)
    
    # Ambil bagian paling keras
    if len(y) > target_len:
        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        max_idx = np.argmax(rms) * HOP_LENGTH
        start = max(0, max_idx - target_len // 2)
        if start + target_len > len(y):
            start = len(y) - target_len
        y = y[start:start+target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
        
    S = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=0, fmax=4000,
        window='hamming', center=False
    )
    return np.log(S + 1e-9)

def main():
    base_dir = r"C:\Users\ASUS\Videos\DATASET"
    categories = ['AMBULANCE', 'FIRETRUCK', 'POLICE', 'NORMAL']
    
    plt.figure(figsize=(15, 10))
    plt.suptitle("Analisis Visual AI (Mel-Spectrogram 40-band)", fontsize=16)
    
    for i, cat in enumerate(categories):
        cat_dir = os.path.join(base_dir, cat)
        if not os.path.exists(cat_dir):
            continue
            
        files = [f for f in os.listdir(cat_dir) if f.endswith('.wav') and 'augment' not in f]
        if not files:
            continue
            
        sample_file = os.path.join(cat_dir, random.choice(files))
        melspec = extract_melspec(sample_file)
        
        plt.subplot(2, 2, i+1)
        librosa.display.specshow(melspec, x_axis='time', y_axis='mel', 
                                 sr=SAMPLE_RATE, hop_length=HOP_LENGTH, fmax=4000)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"{cat} ({os.path.basename(sample_file)})")
        
    plt.tight_layout()
    output_path = r"C:\Users\ASUS\.gemini\antigravity\brain\36105c52-8459-4796-a547-7c54aea5e73e\spectrogram_analysis.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    main()
