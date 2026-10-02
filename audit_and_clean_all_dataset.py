import os
import sys
import glob
import shutil
import numpy as np
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import tensorflow as tf
except ImportError:
    import tflite_runtime.interpreter as tf

# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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

def extract_dsp_single(y):
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
    hamming = np.hamming(N_FFT)
    mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000)

    log_mel_frames = []
    for frame in range(n_frames):
        start = frame * HOP_LENGTH
        frame_data = y_smoothed[start:start+N_FFT].copy()
        frame_data -= np.mean(frame_data)
        frame_win = frame_data * hamming
        fft_complex = np.fft.rfft(frame_win, n=N_FFT)
        power_spec = np.abs(fft_complex) ** 2
        mel_energies = np.dot(mel_fb, power_spec)
        log_mel = np.log(mel_energies + 1e-9)
        log_mel_frames.append(log_mel)

    return np.array(log_mel_frames, dtype=np.float32)

def audit_file_worker(fp, cat):
    expected_cls = FOLDER_TO_CLS[cat]
    fname = os.path.basename(fp)
    try:
        y, sr = librosa.load(fp, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        return {"type": "corrupt", "file": fp, "reason": str(e), "cat": cat}

    rms = np.sqrt(np.mean(y**2))
    if rms < 0.0008 or len(y) < 1000:
        return {"type": "silent", "file": fp, "rms": rms, "cat": cat}

    # Extract DSP
    feat = extract_dsp_single(y)
    return {"type": "valid", "file": fp, "feat": feat, "cat": cat, "expected_cls": expected_cls}

def run_parallel_audit(auto_move=False):
    print("=" * 70)
    print("  AUDIT MENYELURUH DATASET (PARALLEL MULTI-CORE)")
    print(f"  Model AI: {MODEL_PATH}")
    print(f"  Mode: {'MEMINDAHKAN FILE' if auto_move else 'SIMULASI SCAN (Dry-run)'}")
    print("=" * 70)

    # 1. Collect all files
    all_tasks = []
    for cat in ['AMBULANCE', 'FIRETRUCK', 'POLICE']:
        cat_dir = os.path.join(ROOT, cat)
        files = glob.glob(os.path.join(cat_dir, "*.wav")) + glob.glob(os.path.join(cat_dir, "*.ogg"))
        for f in files:
            all_tasks.append((f, cat))

    print(f"[*] Total file yang akan diaudit: {len(all_tasks)} file...")

    # 2. Extract features in parallel
    max_workers = min(12, os.cpu_count() or 4)
    print(f"[*] Menjalankan ekstraksi sinyal pada {max_workers} worker CPU...")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(audit_file_worker, fp, cat): fp for fp, cat in all_tasks}
        completed = 0
        total = len(all_tasks)
        for fut in as_completed(futures):
            completed += 1
            if completed % 300 == 0 or completed == total:
                print(f"    Selesai membaca {completed}/{total} audio...", flush=True)
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                pass

    # 3. Batch Inference on TFLite Model
    print("\n[*] Menjalankan Klasifikasi AI TFLite...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    inp_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    stats = {
        "scanned": len(results),
        "silent_corrupt": [],
        "not_siren": [],
        "salah_kamar": [],
        "clean": 0
    }

    for item in results:
        t = item["type"]
        fp = item["file"]
        cat = item["cat"]
        fname = os.path.basename(fp)

        if t in ["corrupt", "silent"]:
            stats["silent_corrupt"].append(item)
            if auto_move:
                dest = os.path.join(TRASH_DIR, "hening_rusak", fname)
                shutil.move(fp, dest)
            continue

        feat = item["feat"]
        expected_cls = item["expected_cls"]

        inp_shape = inp_details['shape']
        inp_type = inp_details['dtype']
        inp_data = np.expand_dims(feat, axis=(0, -1)) if len(inp_shape) == 4 else np.expand_dims(feat, axis=0)

        if inp_type == np.int8:
            scale, zero_point = inp_details['quantization']
            inp_data = (inp_data / scale + zero_point).clip(-128, 127).astype(np.int8)

        interpreter.set_tensor(inp_details['index'], inp_data)
        interpreter.invoke()
        out = interpreter.get_tensor(out_details['index'])[0]

        if out_details['dtype'] == np.int8:
            scale, zero_point = out_details['quantization']
            out = (out.astype(np.float32) - zero_point) * scale

        if np.max(out) > 1.0 or np.min(out) < 0.0 or abs(np.sum(out) - 1.0) > 0.05:
            exp_s = np.exp(out - np.max(out))
            probs = exp_s / np.sum(exp_s)
        else:
            probs = out

        pred_cls = int(np.argmax(probs))
        pred_conf = float(probs[pred_cls])
        norm_conf = float(probs[2])

        # Kasus 1: Bukan sirine sama sekali (Normal murni >= 75%)
        if pred_cls == 2 and norm_conf >= 0.75 and (probs[0] + probs[1] + probs[3]) < 0.25:
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

        # Kasus 2: Salah kamar dominan (Prediksi kelas sirine lain >= 75% dan unggul telak >= 30% dari kelas aslinya)
        if pred_cls != expected_cls and pred_cls != 2 and pred_conf >= 0.75 and (pred_conf - probs[expected_cls]) >= 0.30:
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
    print("  HASIL AUDIT DATASET SIRENMASTER")
    print("=" * 70)
    print(f"Total File Terpindai       : {stats['scanned']} file")
    print(f"File Bersih (Valid Sirine) : {stats['clean']} file")
    print(f"File Hening / Rusak        : {len(stats['silent_corrupt'])} file")
    print(f"Bukan Sirine (Noise Murni) : {len(stats['not_siren'])} file")
    print(f"Salah Kamar Dominan (>75%) : {len(stats['salah_kamar'])} file")

    if stats["not_siren"]:
        print(f"\n[!] Contoh Suara BUKAN SIRINE di Folder Sirine (Total {len(stats['not_siren'])} file):")
        for it in stats["not_siren"][:10]:
            print(f"    - [{it['expected']}] {os.path.basename(it['file'])} -> Normal: {it['norm_conf']*100:.1f}%")

    if stats["salah_kamar"]:
        print(f"\n[!] Contoh Suara SALAH KAMAR (Total {len(stats['salah_kamar'])} file):")
        for it in stats["salah_kamar"][:15]:
            print(f"    - Folder Asal: {it['expected']:10s} -> AI Deteksi: {it['predicted']:10s} ({it['pred_conf']*100:.1f}%) | File: {os.path.basename(it['file'])}")

    # Write audit log
    log_path = os.path.join(ROOT, "laporan_audit_dataset_terbaru.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("LAPORAN AUDIT MENYELURUH DATASET SIRENMASTER\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Scan      : {stats['scanned']}\n")
        f.write(f"Bersih          : {stats['clean']}\n")
        f.write(f"Hening/Rusak    : {len(stats['silent_corrupt'])}\n")
        f.write(f"Bukan Sirine    : {len(stats['not_siren'])}\n")
        f.write(f"Salah Kamar     : {len(stats['salah_kamar'])}\n\n")

        f.write("DAFTAR BUKAN SIRINE:\n")
        for it in stats["not_siren"]:
            f.write(f"[{it['expected']}] {it['file']} (Normal: {it['norm_conf']*100:.1f}%)\n")

        f.write("\nDAFTAR SALAH KAMAR:\n")
        for it in stats["salah_kamar"]:
            f.write(f"[{it['expected']} -> {it['predicted']} ({it['pred_conf']*100:.1f}%)] {it['file']}\n")

    print(f"\n[+] Log audit tersimpan di: {log_path}")
    return stats

if __name__ == "__main__":
    auto = ("--move" in sys.argv or "-m" in sys.argv)
    run_parallel_audit(auto_move=auto)
