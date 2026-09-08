import os
import sys
import numpy as np
import librosa
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MODEL_PATH = "siren_classifier_model.h5"
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
SAMPLE_RATE = 8000
DURATION = 4.0

def extract_melspec(y, sr=SAMPLE_RATE):
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
        
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y * min(1.0 / max_val, 10.0)
        
    y_smoothed = np.convolve(y, [1/3, 1/3, 1/3], mode='same')
    
    n_frames = (target_len - 256) // 128 + 1
    hamming = np.hamming(256)
    mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=256, n_mels=40, fmin=0, fmax=4000)
    
    log_mel_frames = []
    for frame in range(n_frames):
        start = frame * 128
        frame_data = y_smoothed[start:start+256].copy()
        frame_data -= np.mean(frame_data)
        power_spec = np.abs(np.fft.rfft(frame_data * hamming, n=256)) ** 2
        log_mel_frames.append(np.log(np.dot(mel_fb, power_spec) + 1e-9))
        
    return np.array(log_mel_frames, dtype=np.float32)

print("Memuat Model AI Baru (Keras)...")
model = tf.keras.models.load_model(MODEL_PATH)

print("\n--- UJI COBA KETAJAMAN AI ---")

test_files = [
    ("Suara Sirene Sintesis (Ambulance)", "AMBULANCE/SYNTH_amb_0.wav"),
    ("Suara Bising / Manusia (Normal)", glob.glob("NORMAL/hard_neg_*.wav")[0] if glob.glob("NORMAL/hard_neg_*.wav") else "NORMAL/ESC50_Noise_1-100032-A-0.wav")
]

for name, path in test_files:
    if os.path.exists(path):
        print(f"\n[?] Menguji: {name}")
        y, _ = librosa.load(path, sr=SAMPLE_RATE)
        features = extract_melspec(y)
        
        # Load normalization constants
        mean_arr = np.mean(features, axis=0) # simplified fallback if we don't parse model.h
        
        # Parse model.h for correct mean/std
        import re
        with open("sirenmaster_main/model.h", 'r') as f:
            content = f.read()
        mean_match = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\}', content)
        std_match  = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\}', content)
        MEL_MEAN = np.array([float(x.strip().rstrip('f')) for x in mean_match.group(1).split(',')])
        MEL_STD  = np.array([float(x.strip().rstrip('f')) for x in std_match.group(1).split(',')])
        
        features = (features - MEL_MEAN) / MEL_STD
        features = np.expand_dims(features, axis=(0, -1))
        
        pred = model.predict(features, verbose=0)[0]
        best_idx = np.argmax(pred)
        
        print(f"    -> AI Menjawab: {CATEGORIES[best_idx]} (Keyakinan: {pred[best_idx]*100:.1f}%)")
        print(f"    -> Detail Skor : AMB: {pred[0]*100:.1f}%, FIRE: {pred[1]*100:.1f}%, NORMAL: {pred[2]*100:.1f}%, POL: {pred[3]*100:.1f}%")
    else:
        print(f"\nFile {path} tidak ditemukan untuk diuji.")

print("\nUji Coba Selesai!")
