"""
AUDIO QUALITY AUDITOR - Menggunakan Model AI untuk Mendeteksi File Palsu (Versi Diperbaiki)
Memindai semua file sirine dan memindahkan yang terdeteksi sebagai NORMAL (bukan sirine) atau SILENT ke TRASH.
"""
import os
import sys
import numpy as np
import shutil
import tensorflow as tf
import librosa
import traceback

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATASET_BASE = r"C:\Users\ASUS\Videos\DATASET"
SIREN_CATS = ['AMBULANCE', 'FIRETRUCK', 'POLICE']
LABEL_NAMES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
TRASH_DIR = os.path.join(DATASET_BASE, "TRASH")

# Parameter DSP (Harus 100% cocok dengan model & test_laptop.py)
SAMPLE_RATE = 8000
DURATION = 4.0
N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40
N_FMAX = 4000
BUFFER_SAMPLES = int(SAMPLE_RATE * DURATION)

# Load model TFLite
MODEL_PATH = os.path.join(DATASET_BASE, "siren_model_quant.tflite")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(DATASET_BASE, "siren_classifier_model.tflite")

# Load scaler
scaler = np.load(os.path.join(DATASET_BASE, "siren_scaler.npz"))
global_mean = scaler['global_mean']
global_std = scaler['global_std']

# Pre-compute filterbank & window
_hamming_window = np.hamming(N_FFT).astype(np.float32)
_mel_filterbank = librosa.filters.mel(
    sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=N_FMAX
)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def extract_melspec(y):
    target_len = BUFFER_SAMPLES
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]

    # 3-tap moving average LPF
    y_smoothed = np.convolve(y, [1/3, 1/3, 1/3], mode='same')

    n_frames = (target_len - N_FFT) // HOP_LENGTH + 1
    log_mel_frames = np.empty((n_frames, N_MELS), dtype=np.float32)

    for frame_idx in range(n_frames):
        start = frame_idx * HOP_LENGTH
        frame_data = y_smoothed[start:start + N_FFT].copy()

        # DC removal
        frame_data -= np.mean(frame_data)

        # Hamming windowing
        frame_windowed = frame_data * _hamming_window

        # Power spectrum
        fft_complex = np.fft.rfft(frame_windowed, n=N_FFT)
        power_spec = np.abs(fft_complex) ** 2

        # Mel energies
        mel_energies = np.dot(_mel_filterbank, power_spec)

        # Log energy
        log_mel_frames[frame_idx] = np.log(mel_energies + 1e-9)

    return log_mel_frames

def predict_file(filepath):
    """Prediksi kategori file audio menggunakan model TFLite"""
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        
        # Cek RMS untuk mendeteksi kesunyian
        rms = np.sqrt(np.mean(y**2))
        if rms < 0.0015:
            return "SILENT", 1.0, rms
        
        # Extract Mel Spectrogram
        feat = extract_melspec(y)
        
        # Z-score normalize
        feat = (feat - global_mean) / (global_std + 1e-8)
        
        # Reshape for model input
        feat = feat.reshape(1, 249, 40, 1).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], feat)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        
        pred_idx = np.argmax(output)
        pred_conf = float(output[pred_idx])
        pred_label = LABEL_NAMES[pred_idx]
        
        return pred_label, pred_conf, rms
        
    except Exception as e:
        print(f"\n[ERROR] Gagal memproses file {os.path.basename(filepath)}:")
        traceback.print_exc()
        return "ERROR", 0.0, 0.0

def main():
    print("=" * 65)
    print("  AUDIO QUALITY AUDITOR (AI-Powered) - FIXED VERSION")
    print("=" * 65)
    print(f"[*] Model: {os.path.basename(MODEL_PATH)}")
    print(f"[*] Input shape model: {input_details[0]['shape']}")
    print(f"[*] Direktori TRASH: {TRASH_DIR}")
    os.makedirs(TRASH_DIR, exist_ok=True)
    
    total_scanned = 0
    total_suspect = 0
    total_errors = 0
    suspect_list = []
    
    # Kumpulkan semua file per kategori
    all_files = {}
    for cat in SIREN_CATS:
        cat_path = os.path.join(DATASET_BASE, cat)
        if os.path.exists(cat_path):
            files = sorted([f for f in os.listdir(cat_path) if f.endswith('.wav')])
            all_files[cat] = files
            print(f"[*] Terdeteksi {len(files)} file di folder {cat}")
        else:
            all_files[cat] = []
            
    print("-" * 65)
    
    # Jalankan pemeriksaan
    for cat, files in all_files.items():
        if not files:
            continue
            
        print(f"\n[SCANNING] Kategori {cat} ({len(files)} file)...")
        cat_path = os.path.join(DATASET_BASE, cat)
        cat_suspect = 0
        
        for idx, f in enumerate(files):
            f_path = os.path.join(cat_path, f)
            pred_label, pred_conf, rms = predict_file(f_path)
            total_scanned += 1
            
            # Pengaman awal: Jika ada error beruntun, hentikan script agar tidak merusak dataset
            if pred_label == "ERROR":
                total_errors += 1
                if total_errors >= 10:
                    print("\n[FATAL] Terlalu banyak error berturut-turut! Script dihentikan untuk menjaga keamanan dataset.")
                    sys.exit(1)
                continue
            
            is_suspect = False
            reason = ""
            
            # Aturan file palsu / buruk:
            # 1. Model yakin ini NORMAL (bukan sirine) dengan keyakinan > 75%
            # 2. File terlalu sepi (SILENT)
            # 3. RMS sangat rendah (RMS < 0.002)
            if pred_label == "NORMAL" and pred_conf > 0.75:
                is_suspect = True
                reason = f"AI bilang NORMAL ({pred_conf*100:.1f}%)"
            elif pred_label == "SILENT":
                is_suspect = True
                reason = "File hening (SILENT)"
            elif rms < 0.002:
                is_suspect = True
                reason = f"Volume terlalu rendah (RMS={rms:.5f})"
                
            if is_suspect:
                cat_suspect += 1
                total_suspect += 1
                suspect_list.append((cat, f, reason))
                
                # Pindahkan ke TRASH dengan prefix
                dest = os.path.join(TRASH_DIR, f"FAKE_{cat}_{f}")
                try:
                    shutil.move(f_path, dest)
                    print(f"  [PALSU] {f} -> {reason}")
                except Exception as move_err:
                    print(f"  [ERROR MOVE] Gagal memindahkan {f}: {move_err}")
            
            # Progress print
            if (idx + 1) % 200 == 0:
                print(f"  ... memproses {idx+1}/{len(files)} file")
                
        print(f"  [{cat}] Selesai! File palsu dipindahkan: {cat_suspect}/{len(files)}")

    print("\n" + "=" * 65)
    print("  LAPORAN AKHIR AUDIT KUALITAS AUDIO")
    print("=" * 65)
    print(f"Total file dipindai       : {total_scanned}")
    print(f"Total file PALSU dipindah : {total_suspect}")
    print(f"Total file BERSIH tersisa : {total_scanned - total_suspect - total_errors}")
    print(f"Total file ERROR          : {total_errors}")
    print("-" * 65)
    
    if suspect_list:
        print("\nContoh beberapa file yang dipindahkan ke TRASH:")
        for cat, fname, reason in suspect_list[:30]:
            print(f"  [{cat}] {fname} -> {reason}")
        if len(suspect_list) > 30:
            print(f"  ... dan {len(suspect_list) - 30} file lainnya.")
    else:
        print("\nLuar biasa! Tidak ada file palsu yang terdeteksi.")
        
    print("\nSemua file palsu aman dipindahkan ke folder TRASH.")
    print("AUDIT KUALITAS SELESAI!")

if __name__ == "__main__":
    main()
