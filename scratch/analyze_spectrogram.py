import os
import numpy as np
import librosa

SAMPLE_RATE = 8000
DURATION = 4.0
BUFFER_SAMPLES = int(SAMPLE_RATE * DURATION)
N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40
N_FMAX = 4000

# Pre-compute filterbank & window
_hamming_window = np.hamming(N_FFT).astype(np.float32)
_mel_filterbank = librosa.filters.mel(
    sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=N_FMAX
)

def extract_melspec(y):
    y_smoothed = np.convolve(y, [1/3, 1/3, 1/3], mode='same')
    n_frames = (len(y) - N_FFT) // HOP_LENGTH + 1
    log_mel_frames = np.empty((n_frames, N_MELS), dtype=np.float32)
    for frame_idx in range(n_frames):
        start = frame_idx * HOP_LENGTH
        frame_data = y_smoothed[start:start + N_FFT].copy()
        frame_data -= np.mean(frame_data)
        frame_windowed = frame_data * _hamming_window
        fft_complex = np.fft.rfft(frame_windowed, n=N_FFT)
        power_spec = np.abs(fft_complex) ** 2
        mel_energies = np.dot(_mel_filterbank, power_spec)
        log_mel_frames[frame_idx] = np.log(mel_energies + 1e-9)
    return log_mel_frames

# Load recorded resampled audio
rec_path = "scratch/recorded_resampled.wav"
y_rec, _ = librosa.load(rec_path, sr=SAMPLE_RATE)
feat_rec = extract_melspec(y_rec)

# Find a real POLICE/AMBULANCE training file to compare
import glob
real_files = glob.glob(r"C:\Users\ASUS\Videos\DATASET\POLICE\*.wav")
if real_files:
    real_path = real_files[0]
    y_real, _ = librosa.load(real_path, sr=SAMPLE_RATE)
    y_real = y_real[:BUFFER_SAMPLES]
    if len(y_real) < BUFFER_SAMPLES:
        y_real = np.pad(y_real, (0, BUFFER_SAMPLES - len(y_real)), mode='constant')
    feat_real = extract_melspec(y_real)
    
    print("=== SPECTRAL COMPARISON (Mean energy per mel bin) ===")
    mean_rec = np.mean(feat_rec, axis=0)
    mean_real = np.mean(feat_real, axis=0)
    
    print(f"Mel Bin | Recorded Mic | Real Training | Diff")
    print("-" * 50)
    for i in range(N_MELS):
        print(f"  {i:2d}    |   {mean_rec[i]:7.2f}    |   {mean_real[i]:7.2f}     | {mean_rec[i]-mean_real[i]:6.2f}")
    
    print("\nOverall Stats:")
    print(f"  Recorded Mic - Min: {np.min(mean_rec):.2f}, Max: {np.max(mean_rec):.2f}, Mean: {np.mean(mean_rec):.2f}")
    print(f"  Real Training - Min: {np.min(mean_real):.2f}, Max: {np.max(mean_real):.2f}, Mean: {np.mean(mean_real):.2f}")
else:
    print("No training files found to compare.")
