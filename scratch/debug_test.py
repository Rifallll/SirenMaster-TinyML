import os
import numpy as np
import librosa
import tensorflow as tf

SAMPLE_RATE = 8000
DURATION = 4.0
N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40
N_FMAX = 4000
BUFFER_SAMPLES = int(SAMPLE_RATE * DURATION)
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

CACHE_PATH = "siren_40x249_melspec_cache_lokal.npz"
MODEL_PATH = "siren_model_quant.tflite"

# Pre-compute filterbank & window
_hamming_window = np.hamming(N_FFT).astype(np.float32)
_mel_filterbank = librosa.filters.mel(
    sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=N_FMAX
)

def extract_melspec(y):
    target_len = BUFFER_SAMPLES
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]

    y_smoothed = np.convolve(y, [1/3, 1/3, 1/3], mode='same')
    n_frames = (target_len - N_FFT) // HOP_LENGTH + 1
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

# Load scaler stats
with np.load(CACHE_PATH, allow_pickle=True) as data:
    X_train = data['X_train']
    global_mean = np.mean(X_train, axis=(0, 1))
    global_std = np.std(X_train, axis=(0, 1))
    global_std[global_std < 1e-6] = 1.0

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    # Take first 4 seconds
    y = y[:BUFFER_SAMPLES]
    if len(y) < BUFFER_SAMPLES:
        y = np.pad(y, (0, BUFFER_SAMPLES - len(y)), mode='constant')
    
    # Auto-gain (same as test_laptop.py)
    max_val = np.max(np.abs(y))
    if max_val > 0.0001:
        gain = min(1.0 / max_val, 5.0)
        y = y * gain

    feat = extract_melspec(y)
    feat_scaled = (feat - global_mean) / global_std
    feat_scaled = feat_scaled.astype(np.float32)
    feat_scaled = np.expand_dims(feat_scaled, axis=0)
    feat_scaled = np.expand_dims(feat_scaled, axis=-1)

    input_scale, input_zero_point = input_details[0]['quantization']
    if input_details[0]['dtype'] == np.int8:
        feat_quant = np.round(feat_scaled / input_scale) + input_zero_point
        feat_quant = np.clip(feat_quant, -128, 127).astype(np.int8)
    else:
        feat_quant = feat_scaled

    interpreter.set_tensor(input_details[0]['index'], feat_quant)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_scale, output_zero_point = output_details[0]['quantization']
    if output_details[0]['dtype'] == np.int8:
        probs = (output_data[0].astype(np.float32) - output_zero_point) * output_scale
    else:
        probs = output_data[0]
        
    probs_clean = np.clip(probs, 0.0, None)
    total = np.sum(probs_clean)
    if total > 0:
        probs_norm = probs_clean / total
    else:
        probs_norm = np.array([0.0, 0.0, 1.0, 0.0])
    return probs_norm

# Test on 2 random files from each category
import glob, random
for cat in CATEGORIES:
    cat_dir = os.path.join(r"C:\Users\ASUS\Videos\DATASET", cat)
    files = glob.glob(os.path.join(cat_dir, "*.wav"))
    if not files:
        print(f"No files found in {cat}")
        continue
    test_files = random.sample(files, min(2, len(files)))
    print(f"\n=== Category: {cat} ===")
    for fp in test_files:
        probs = predict_audio_file(fp)
        pred = CATEGORIES[np.argmax(probs)]
        print(f"  File: {os.path.basename(fp)}")
        print(f"    Prediction: {pred} | Probs: {np.round(probs, 2)}")
