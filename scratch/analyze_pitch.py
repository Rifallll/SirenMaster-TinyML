import librosa
import numpy as np

SAMPLE_RATE = 8000
rec_path = "scratch/recorded_resampled.wav"
y, _ = librosa.load(rec_path, sr=SAMPLE_RATE)

print(f"Analyzing dominant frequencies of {rec_path} over time...")
print(f"Total samples: {len(y)}, duration: {len(y)/SAMPLE_RATE}s")

# Let's divide into 8 frames (0.5s each)
frame_len = 4000
for i in range(8):
    start = i * frame_len
    chunk = y[start:start+frame_len]
    if len(chunk) < frame_len:
        break
        
    # Compute FFT
    fft_data = np.fft.rfft(chunk)
    freqs = np.fft.rfftfreq(len(chunk), d=1/SAMPLE_RATE)
    mags = np.abs(fft_data)
    
    # Get top 3 peaks
    top_indices = np.argsort(mags)[-5:][::-1]
    peak_freqs = freqs[top_indices]
    peak_mags = mags[top_indices]
    
    print(f"Frame {i} ({i*0.5:.1f}s - {(i+1)*0.5:.1f}s):")
    print(f"  Top peaks: " + ", ".join([f"{f:.1f}Hz ({m:.2f})" for f, m in zip(peak_freqs, peak_mags)]))
