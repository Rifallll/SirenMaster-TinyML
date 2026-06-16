import os
import numpy as np
import librosa
from scipy.signal import butter, lfilter
import tensorflow as tf

SAMPLE_RATE = 8000
DURATION = 4.0
BUFFER_SAMPLES = int(SAMPLE_RATE * DURATION)
N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40
N_FMAX = 4000
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

MODEL_PATH = "siren_model_quant.tflite"
CACHE_PATH = "siren_40x249_melspec_cache_lokal.npz"

_hamming_window = np.hamming(N_FFT).astype(np.float32)
_mel_filterbank = librosa.filters.mel(
    sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=N_FMAX
)

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

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

# Load scaler stats
with np.load(CACHE_PATH, allow_pickle=True) as data:
    X_train = data['X_train']
    global_mean = np.mean(X_train, axis=(0, 1))
    global_std = np.std(X_train, axis=(0, 1))
    global_std[global_std < 1e-6] = 1.0

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(y):
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
    return probs_clean / total if total > 0 else np.array([0.0, 0.0, 1.0, 0.0])

# Load recording
rec_path = "scratch/recorded_resampled.wav"
y, _ = librosa.load(rec_path, sr=SAMPLE_RATE)

# Try different High-Pass filter cutoffs
for cutoff in [0, 150, 300, 450]:
    if cutoff == 0:
        y_filt = y
    else:
        y_filt = highpass_filter(y, cutoff, SAMPLE_RATE)
        
    # Auto-gain
    max_val = np.max(np.abs(y_filt))
    if max_val > 0.0001:
        y_filt = y_filt / max_val
        
    probs = predict(y_filt)
    pred = CATEGORIES[np.argmax(probs)]
    print(f"Cutoff: {cutoff:3d} Hz -> Pred: {pred:10s} | Probs: {np.round(probs, 2)}")
