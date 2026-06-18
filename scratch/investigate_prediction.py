import os
import numpy as np
import tensorflow as tf
import librosa
import re

DATASET_BASE = r"C:\Users\ASUS\Videos\DATASET"
MODEL_H = os.path.join(DATASET_BASE, "sirenmaster_main", "model.h")
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

# Load scaling stats
with open(MODEL_H, 'r') as f:
    content = f.read()
mean_match = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\}', content)
std_match  = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\}', content)
MEL_MEAN = np.array([float(x.strip().rstrip('f')) for x in mean_match.group(1).split(',')])
MEL_STD  = np.array([float(x.strip().rstrip('f')) for x in std_match.group(1).split(',')])

# Load Keras model
keras_model = tf.keras.models.load_model(os.path.join(DATASET_BASE, "siren_classifier_model.h5"))

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=os.path.join(DATASET_BASE, "siren_model_quant.tflite"))
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

def extract_features(fpath):
    y, _ = librosa.load(fpath, sr=8000, mono=True, duration=4.0)
    if len(y) < 32000:
        repeats = int(np.ceil(32000 / len(y)))
        y = np.tile(y, repeats)[:32000]
    else:
        y = y[:32000]
    y = np.convolve(y, [1/3, 1/3, 1/3], mode='same')
    
    n_frames = (32000 - 256) // 128 + 1
    hamming = np.hamming(256)
    mel_fb  = librosa.filters.mel(sr=8000, n_fft=256, n_mels=40, fmin=0, fmax=4000)
    
    log_mel_frames = []
    for i in range(n_frames):
        start = i * 128
        frame = y[start:start+256].copy()
        frame -= np.mean(frame)
        frame *= hamming
        fft   = np.fft.rfft(frame, n=256)
        power = np.abs(fft)**2
        mel   = mel_fb @ power
        log_mel_frames.append(np.log(mel + 1e-9))
    spec = np.array(log_mel_frames, dtype=np.float32)
    spec = (spec - MEL_MEAN) / (MEL_STD + 1e-8)
    return spec

def predict_keras(spec):
    inp = spec[np.newaxis, ..., np.newaxis]
    probs = keras_model.predict(inp, verbose=0)[0]
    return probs

def predict_tflite(spec):
    inp = spec[np.newaxis, ..., np.newaxis]
    sc = inp_det['quantization_parameters']['scales'][0]
    zp = inp_det['quantization_parameters']['zero_points'][0]
    inp_q = np.clip(np.round(inp / sc) + zp, -128, 127).astype(np.int8)
    interpreter.set_tensor(inp_det['index'], inp_q)
    interpreter.invoke()
    out = interpreter.get_tensor(out_det['index'])[0].astype(np.float32)
    sc2 = out_det['quantization_parameters']['scales'][0]
    zp2 = out_det['quantization_parameters']['zero_points'][0]
    probs = (out - zp2) * sc2
    return probs

# Test a problematic file
target_file = os.path.join(DATASET_BASE, "NORMAL", "urban_0_162103-0-0-6.wav")
if os.path.exists(target_file):
    spec = extract_features(target_file)
    p_k = predict_keras(spec)
    p_t = predict_tflite(spec)
    print(f"\nFile: {os.path.basename(target_file)}")
    print(f"Keras probs : {list(np.round(p_k, 4))}")
    print(f"TFLite probs: {list(np.round(p_t, 4))}")
else:
    print(f"File not found: {target_file}")
