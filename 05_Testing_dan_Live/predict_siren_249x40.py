import os
import sys
import argparse
import numpy as np
import librosa
import tensorflow as tf

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

# Default parameters matching model.h
SAMPLE_RATE = 8000
N_FFT = 256
HOP_LENGTH = 128
N_MFCC = 13
N_MELS = 40
DURATION = 4.0

def extract_features(y, mean, std):
    """Extracts and normalizes 40x63 Mel Spectrogram features matching the ESP32 DSP exactly."""
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
        
    # NORMALIZE AUDIO (Auto-Gain) dengan batas maksimal
    # Jangan mendongkrak suara terlalu ekstrem, maksimal 10x lipat
    # agar suara desisan kipas angin tidak ikut jadi keras dan dianggap Firetruck.
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        gain = min(1.0 / max_val, 10.0)
        y = y * gain
        
    # Gunakan Melspectrogram persis seperti di C++, BUKAN MFCC
    # center=False sangat penting agar tidak ada padding (menghasilkan persis 63 frame, bukan 65)
    melspec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, 
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=0, fmax=4000,
        window='hamming', center=False
    )
    
    # Terapkan log (meniru logf(energy + 1e-9f) di C++)
    log_mel = np.log(melspec + 1e-9)
    
    # Shape: (63, 40)
    feat = log_mel.T
    
    # Scale feature using training mean and standard deviation
    feat_scaled = (feat - mean[np.newaxis, :]) / std[np.newaxis, :]
    return feat_scaled

def run_tflite_inference(model_path, feat_scaled):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check if input tensor is INT8 quantized
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        # Quantize float input to int8
        feat_quant = np.round(feat_scaled / scale) + zero_point
        feat_input = np.clip(feat_quant, -128, 127).astype(np.int8)
    else:
        feat_input = feat_scaled.astype(np.float32)
        
    # Model CNN expects shape 4D e.g., (1, 63, 40, 1)
    feat_input = np.expand_dims(feat_input, axis=0)
    feat_input = np.expand_dims(feat_input, axis=-1)
    
    interpreter.set_tensor(input_details[0]['index'], feat_input)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Check if output tensor is INT8 quantized
    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        # Dequantize int8 output to float
        probs = (output_data[0].astype(np.float32) - zero_point) * scale
    else:
        probs = output_data[0]
        
    return probs

def main():
    parser = argparse.ArgumentParser(description="Predict Emergency Vehicle Siren type from audio WAV file (13x63).")
    parser.add_argument('--file', type=str, required=True, help="Path to the WAV audio file.")
    parser.add_argument('--model', type=str, default='siren_model_quant.tflite', help="Path to quantized TFLite model.")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: WAV file not found: {args.file}")
        sys.exit(1)
        
    # Read mean and std parameters from generated model.h to match scaler exactly
    model_h_path = r"siren_detection\model.h"
    if not os.path.exists(model_h_path):
        print(f"Error: Deploy model.h first or create it at {model_h_path}")
        sys.exit(1)
        
    # Parse mean and std from model.h
    mean = []
    std = []
    with open(model_h_path, 'r') as f:
        content = f.read()
        mean_match = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\};', content)
        std_match = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\};', content)
        
        if mean_match and std_match:
            mean = np.array([float(x.strip().replace('f', '')) for x in mean_match.group(1).split(',')])
            std = np.array([float(x.strip().replace('f', '')) for x in std_match.group(1).split(',')])
        else:
            print("Error: Could not parse MFCC scaling parameters from model.h")
            sys.exit(1)
            
    print(f"[*] Processing file: {args.file}")
    print(f"[*] Loaded scaler mean: {np.round(mean, 2)}")
    print(f"[*] Loaded scaler std:  {np.round(std, 2)}")
    
    # Load audio
    y, sr = librosa.load(args.file, sr=SAMPLE_RATE)
    
    # Slice sliding windows (1.024s chunks, 50% overlap)
    target_len = int(SAMPLE_RATE * DURATION)
    hop = target_len // 2
    
    chunks = []
    for start in range(0, len(y) - target_len + 1, hop):
        chunk = y[start:start+target_len]
        rms = np.sqrt(np.mean(chunk**2))
        if rms >= 0.001: # Skip silence
            chunks.append(chunk)
            
    if len(chunks) == 0:
        if len(y) >= target_len // 2:
            padded = np.pad(y, (0, target_len - len(y)), mode='constant')
            chunks.append(padded)
        else:
            print("Error: Audio is too short or silent.")
            sys.exit(1)
            
    # Load model and run prediction on each chunk
    all_probs = []
    tflite_model_path = args.model
    if not os.path.exists(tflite_model_path):
        # Fallback to look in current folder
        tflite_model_path = 'siren_model_quant.tflite'
        if not os.path.exists(tflite_model_path):
            # Check for any .tflite files in model directory
            tflite_files = [f for f in os.listdir('.') if f.endswith('.tflite')]
            if len(tflite_files) > 0:
                tflite_model_path = tflite_files[0]
            else:
                print("Error: TFLite model file not found.")
                sys.exit(1)
                
    print(f"[*] Using TFLite model: {tflite_model_path}")
    
    for idx, chunk in enumerate(chunks):
        feat = extract_features(chunk, mean, std)
        probs = run_tflite_inference(tflite_model_path, feat)
        all_probs.append(probs)
        
    # Average predictions
    avg_probs = np.mean(all_probs, axis=0)
    
    print("\n" + "="*45)
    print(f" {'Category':<15} | {'Confidence (%)':<15}")
    print("="*45)
    
    best_idx = np.argmax(avg_probs)
    for idx, (cat, prob) in enumerate(zip(CATEGORIES, avg_probs)):
        marker = ">>>" if idx == best_idx else "   "
        print(f"{marker} {cat:<12} | {prob*100:>8.2f}%")
        
    print("="*45)
    print(f"Result: Classified as {CATEGORIES[best_idx]} ({avg_probs[best_idx]*100:.2f}% confidence)\n")

if __name__ == "__main__":
    import re
    main()
