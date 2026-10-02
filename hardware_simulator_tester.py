import os
import sys
import glob
import numpy as np
import librosa

try:
    import tensorflow as tf
except ImportError:
    import tflite_runtime.interpreter as tf

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# --- CONFIGURATION MATCHING FIRMWARE (sirenmaster_main.ino) ---
SAMPLE_RATE = 8000
DURATION = 4.0
N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
SIREN_TRIGGER_THRESHOLD = 0.60  # 60%
NORMAL_THRESHOLD = 0.35

MODEL_PATH = os.path.join(os.path.dirname(__file__), "sirenmaster_main", "model.tflite")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "siren_model_quant.tflite")

def extract_dsp_features(y, sr=SAMPLE_RATE):
    """Exact C++ DSP Pipeline implementation."""
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        repeats = int(np.ceil(target_len / len(y)))
        y = np.tile(y, repeats)[:target_len]
    else:
        y = y[:target_len]

    # Auto-Gain (Max 10x)
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        gain = min(1.0 / max_val, 10.0)
        y = y * gain

    # 1. 3-Tap Moving Average Low Pass Filter
    y_smoothed = np.convolve(y, [1/3, 1/3, 1/3], mode='same')

    # 2. Frame-by-frame STFT with DC removal and Hamming windowing
    n_frames = (target_len - N_FFT) // HOP_LENGTH + 1
    hamming = np.hamming(N_FFT)
    mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000)

    log_mel_frames = []
    for frame in range(n_frames):
        start = frame * HOP_LENGTH
        frame_data = y_smoothed[start:start+N_FFT].copy()
        
        # DC removal
        frame_mean = np.mean(frame_data)
        frame_data = frame_data - frame_mean
        
        # Hamming windowing
        frame_windowed = frame_data * hamming
        
        # Power spectrum
        fft_complex = np.fft.rfft(frame_windowed, n=N_FFT)
        power_spec = np.abs(fft_complex) ** 2
        
        # Mel energies
        mel_energies = np.dot(mel_fb, power_spec)
        
        # Log energy (1e-9 floor)
        log_mel = np.log(mel_energies + 1e-9)
        log_mel_frames.append(log_mel)

    return np.array(log_mel_frames, dtype=np.float32) # (249, 40)

class HardwareSimulator:
    def __init__(self, model_path=MODEL_PATH):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.ema_probs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self.locked_class = 2  # Default NORMAL
        self.alert_streak = 0
        
    def reset_state(self):
        self.ema_probs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self.locked_class = 2
        self.alert_streak = 0

    def run_inference(self, melspec):
        # Format input (1, 249, 40) or (1, 249, 40, 1) depending on model shape
        inp_shape = self.input_details[0]['shape']
        inp_type = self.input_details[0]['dtype']
        
        if len(inp_shape) == 4:
            inp_data = np.expand_dims(melspec, axis=(0, -1))
        else:
            inp_data = np.expand_dims(melspec, axis=0)

        # Handle INT8 Quantization if applicable
        if inp_type == np.int8:
            scale, zero_point = self.input_details[0]['quantization']
            inp_data = (inp_data / scale + zero_point).clip(-128, 127).astype(np.int8)

        self.interpreter.set_tensor(self.input_details[0]['index'], inp_data)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Dequantize if output is INT8
        if self.output_details[0]['dtype'] == np.int8:
            scale, zero_point = self.output_details[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale

        # Softmax if not normalized
        if np.max(output) > 1.0 or np.min(output) < 0.0 or abs(np.sum(output) - 1.0) > 0.05:
            exp_scores = np.exp(output - np.max(output))
            probs = exp_scores / np.sum(exp_scores)
        else:
            probs = output

        # Firmware Exponential Moving Average (EMA) smoothing: alpha = 0.6
        alpha = 0.6
        self.ema_probs = alpha * probs + (1 - alpha) * self.ema_probs

        # Decision Rule matching Firmware logic
        raw_pred = int(np.argmax(self.ema_probs))
        raw_conf = float(self.ema_probs[raw_pred])
        
        # Alert Decision: Siren needs to meet trigger threshold
        siren_mask = [0, 1, 3] # AMB, FIRE, POLICE
        top_siren_idx = siren_mask[np.argmax(self.ema_probs[siren_mask])]
        top_siren_conf = float(self.ema_probs[top_siren_idx])
        normal_conf = float(self.ema_probs[2])

        if top_siren_conf >= SIREN_TRIGGER_THRESHOLD and top_siren_conf > normal_conf:
            final_class = top_siren_idx
            final_conf = top_siren_conf
            status = f"🚨 ALERT ({CATEGORIES[final_class]})"
        else:
            final_class = 2  # NORMAL
            final_conf = normal_conf
            status = "🟢 NORMAL / AMAN"

        return {
            "probs": self.ema_probs.copy(),
            "raw_pred": CATEGORIES[raw_pred],
            "raw_conf": raw_conf,
            "final_class": CATEGORIES[final_class],
            "final_conf": final_conf,
            "status": status
        }

    def test_audio_file(self, file_path):
        self.reset_state()
        print(f"\n[+] Testing Audio: {os.path.basename(file_path)}")
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        target_len = int(SAMPLE_RATE * DURATION)
        
        results = []
        hop = target_len // 2  # 2-second overlap test
        if len(y) < target_len:
            chunks = [y]
        else:
            chunks = [y[s:s+target_len] for s in range(0, len(y) - target_len + 1, hop)]

        for idx, chunk in enumerate(chunks):
            melspec = extract_dsp_features(chunk)
            res = self.run_inference(melspec)
            time_sec = idx * (hop / SAMPLE_RATE)
            p = res['probs']
            print(f"  T={time_sec:4.1f}s | Amb:{p[0]*100:4.1f}% Fir:{p[1]*100:4.1f}% Nor:{p[2]*100:4.1f}% Pol:{p[3]*100:4.1f}% | {res['status']}")
            results.append(res)
            
        return results

def main():
    print("=" * 65)
    print("  TINYML HARDWARE-EQUIVALENT AUDIO SIMULATOR & TESTER")
    print("  Matching ESP32-S2 DSP & Post-Processing Pipeline")
    print("=" * 65)
    
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found: {MODEL_PATH}")
        sys.exit(1)
        
    sim = HardwareSimulator(MODEL_PATH)
    print(f"[*] Model loaded successfully: {MODEL_PATH}")
    
    # Check if specific audio path provided in args
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isfile(target):
            sim.test_audio_file(target)
        elif os.path.isdir(target):
            for ext in ("*.wav", "*.ogg", "*.mp3"):
                for fp in glob.glob(os.path.join(target, ext)):
                    sim.test_audio_file(fp)
    else:
        # Default test against NOISE_TEST folder
        test_dir = os.path.join(os.path.dirname(__file__), "NOISE_TEST")
        if os.path.exists(test_dir):
            print(f"\n[*] Running default hardware simulation test on {test_dir}...")
            files = glob.glob(os.path.join(test_dir, "*.wav")) + glob.glob(os.path.join(test_dir, "*.ogg"))
            for fp in files[:5]:
                sim.test_audio_file(fp)
        else:
            print("[*] Berikan argumen path file audio untuk dites. Contoh:")
            print("    python hardware_simulator_tester.py NOISE_TEST/bending_machine.ogg")

if __name__ == "__main__":
    main()
