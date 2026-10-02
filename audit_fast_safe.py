import os
import sys
import glob
import shutil
import numpy as np
import soundfile as sf
import librosa

# Prevent Numba/LLVM memory issue
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import tensorflow as tf
except ImportError:
    import tflite_runtime.interpreter as tf

ROOT = r"C:\Users\ASUS\Videos\DATASET"
MODEL_PATH = os.path.join(ROOT, "sirenmaster_main", "model.tflite")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT, "siren_model_quant.tflite")

TRASH_DIR = os.path.join(ROOT, "TRASH", "audit_hasil_pemisahan")
os.makedirs(os.path.join(TRASH_DIR, "bukan_sirine"), exist_ok=True)
os.makedirs(os.path.join(TRASH_DIR, "salah_kamar"), exist_ok=True)
os.makedirs(os.path.join(TRASH_DIR, "hening_rusak"), exist_ok=True)

SAMPLE_RATE = 8000
DURATION = 4.0
N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
FOLDER_TO_CLS = {'AMBULANCE': 0, 'FIRETRUCK': 1, 'NORMAL': 2, 'POLICE': 3}

# Pre-compute filterbank & window
HAMMING = np.hamming(N_FFT)
MEL_FB = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000)

def extract_dsp_fast(y, orig_sr):
    if orig_sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
        
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        repeats = int(np.ceil(target_len / len(y)))
        y = np.tile(y, repeats)[:target_len]
    else:
        y = y[:target_len]

    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        gain = min(1.0 / max_val, 10.0)
        y = y * gain

    y_smoothed = np.convolve(y, [1/3, 1/3, 1/3], mode='same')
    n_frames = (target_len - N_FFT) // HOP_LENGTH + 1

    log_mel_frames = []
    for frame in range(n_frames):
        start = frame * HOP_LENGTH
        frame_data = y_smoothed[start:start+N_FFT].copy()
        frame_data -= np.mean(frame_data)
        frame_win = frame_data * HAMMING
        fft_complex = np.fft.rfft(frame_win, n=N_FFT)
        power_spec = np.abs(fft_complex) ** 2
        mel_energies = np.dot(MEL_FB, power_spec)
        log_mel = np.log(mel_energies + 1e-9)
        log_mel_frames.append(log_mel)

    return np.array(log_mel_frames, dtype=np.float32)

def run_safe_audit(auto_move=True):
    print("=" * 70)
    print("  AUDIT MENYELURUH DATASET SIRENMASTER (MEMORY SAFE)")
    print(f"  Model AI: {MODEL_PATH}")
    print(f"  Tindakan Pemisahan: {'AKTIF (Dipindahkan ke TRASH)' if auto_move else 'SIMULASI (Dry Run)'}")
    print("=" * 70)

    # Initialize TFLite
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    inp_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    inp_shape = inp_details['shape']
    inp_type = inp_details['dtype']
    scale_in, zp_in = inp_details['quantization']
    scale_out, zp_out = out_details['quantization']

    stats = {
        "scanned": 0,
        "silent_corrupt": [],
        "not_siren": [],
        "salah_kamar": [],
        "clean": 0
    }

    for cat in ['AMBULANCE', 'FIRETRUCK', 'POLICE']:
        cat_dir = os.path.join(ROOT, cat)
        expected_cls = FOLDER_TO_CLS[cat]
        files = glob.glob(os.path.join(cat_dir, "*.wav")) + glob.glob(os.path.join(cat_dir, "*.ogg"))
        print(f"\n[*] Mengaudit Folder {cat:10s} ({len(files)} file)...")

        for idx, fp in enumerate(files):
            stats["scanned"] += 1
            fname = os.path.basename(fp)
            if (idx + 1) % 400 == 0 or (idx + 1) == len(files):
                print(f"    - Terperiksa {idx + 1}/{len(files)}...", flush=True)

            try:
                y, sr = sf.read(fp)
                if y.ndim > 1:
                    y = np.mean(y, axis=1) # Mono
            except Exception as e:
                try:
                    y, sr = librosa.load(fp, sr=SAMPLE_RATE, mono=True)
                except Exception as e2:
                    stats["silent_corrupt"].append({"file": fp, "reason": str(e2)})
                    if auto_move:
                        shutil.move(fp, os.path.join(TRASH_DIR, "hening_rusak", fname))
                    continue

            rms = np.sqrt(np.mean(y**2))
            if rms < 0.0005 or len(y) < 1000:
                stats["silent_corrupt"].append({"file": fp, "reason": f"RMS Hening ({rms:.6f})"})
                if auto_move:
                    shutil.move(fp, os.path.join(TRASH_DIR, "hening_rusak", fname))
                continue

            # Extract DSP
            feat = extract_dsp_fast(y, sr)
            inp_data = np.expand_dims(feat, axis=(0, -1)) if len(inp_shape) == 4 else np.expand_dims(feat, axis=0)

            if inp_type == np.int8:
                inp_data = (inp_data / scale_in + zp_in).clip(-128, 127).astype(np.int8)

            interpreter.set_tensor(inp_details['index'], inp_data)
            interpreter.invoke()
            out = interpreter.get_tensor(out_details['index'])[0]

            if out_details['dtype'] == np.int8:
                out = (out.astype(np.float32) - zp_out) * scale_out

            if np.max(out) > 1.0 or np.min(out) < 0.0 or abs(np.sum(out) - 1.0) > 0.05:
                exp_s = np.exp(out - np.max(out))
                probs = exp_s / np.sum(exp_s)
            else:
                probs = out

            pred_cls = int(np.argmax(probs))
            pred_conf = float(probs[pred_cls])
            norm_conf = float(probs[2])

            # 1. Bukan Sirine (Dominan Normal >= 80% dan sirine total < 20%)
            if pred_cls == 2 and norm_conf >= 0.80 and (probs[0] + probs[1] + probs[3]) < 0.20:
                stats["not_siren"].append({
                    "file": fp,
                    "expected": cat,
                    "norm_conf": norm_conf,
                    "scores": probs.tolist()
                })
                if auto_move:
                    dest = os.path.join(TRASH_DIR, "bukan_sirine", f"{cat}_{fname}")
                    shutil.move(fp, dest)
                continue

            # 2. Salah Kamar Jelas (Prediksi kelas sirine lain >= 80% dan unggul >= 35% dari kelas aslinya)
            if pred_cls != expected_cls and pred_cls != 2 and pred_conf >= 0.80 and (pred_conf - probs[expected_cls]) >= 0.35:
                stats["salah_kamar"].append({
                    "file": fp,
                    "expected": cat,
                    "predicted": CATEGORIES[pred_cls],
                    "pred_conf": pred_conf,
                    "expected_conf": float(probs[expected_cls]),
                    "scores": probs.tolist()
                })
                if auto_move:
                    dest = os.path.join(TRASH_DIR, "salah_kamar", f"FROM_{cat}_TO_{CATEGORIES[pred_cls]}_{fname}")
                    shutil.move(fp, dest)
                continue

            stats["clean"] += 1

    # Print Summary
    print("\n" + "=" * 70)
    print("  HASIL AUDIT & PEMISAHAN DATASET SELESAI")
    print("=" * 70)
    print(f"Total File Terpindai           : {stats['scanned']} file")
    print(f"File Bersih (Valid Sirine)     : {stats['clean']} file")
    print(f"File Hening / Rusak            : {len(stats['silent_corrupt'])} file (Dipindahkan ke TRASH/hening_rusak)")
    print(f"Bukan Sirine (Noise Murni)     : {len(stats['not_siren'])} file (Dipindahkan ke TRASH/bukan_sirine)")
    print(f"Salah Kamar Dominan (>80%)     : {len(stats['salah_kamar'])} file (Dipindahkan ke TRASH/salah_kamar)")

    # Save log
    log_path = os.path.join(ROOT, "laporan_audit_pemisahan_dataset.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("LAPORAN AUDIT & PEMISAHAN DATASET SIRENMASTER\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Scan   : {stats['scanned']}\n")
        f.write(f"Valid/Bersih : {stats['clean']}\n")
        f.write(f"Hening/Rusak : {len(stats['silent_corrupt'])}\n")
        f.write(f"Bukan Sirine : {len(stats['not_siren'])}\n")
        f.write(f"Salah Kamar  : {len(stats['salah_kamar'])}\n\n")

        f.write("DAFTAR BUKAN SIRINE (DIPISAHKAN):\n")
        for it in stats["not_siren"]:
            f.write(f"[{it['expected']}] {os.path.basename(it['file'])} (Normal: {it['norm_conf']*100:.1f}%)\n")

        f.write("\nDAFTAR SALAH KAMAR (DIPISAHKAN):\n")
        for it in stats["salah_kamar"]:
            f.write(f"[{it['expected']} -> {it['predicted']} ({it['pred_conf']*100:.1f}%)] {os.path.basename(it['file'])}\n")

    print(f"\n[+] Laporan pemisahan dataset tersimpan di: {log_path}")
    return stats

if __name__ == "__main__":
    run_safe_audit(auto_move=True)
