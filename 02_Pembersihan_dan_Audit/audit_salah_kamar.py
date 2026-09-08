import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import glob
import numpy as np
import librosa
try:
    import tensorflow as tf
    tflite = tf.lite
except ImportError:
    import tflite_runtime.interpreter as tflite

ROOT         = r"C:\Users\ASUS\Videos\DATASET"
TFLITE_MODEL = os.path.join(ROOT, "siren_model_quant.tflite")

SAMPLE_RATE = 8000
N_FFT       = 256
HOP_LEN     = 128
N_MELS      = 40
N_FRAMES    = 249

CLASS_NAMES     = {0: "AMBULANCE", 1: "FIRETRUCK", 2: "NORMAL", 3: "POLICE"}
FOLDER_TO_CLASS = {"AMBULANCE": 0, "FIRETRUCK": 1, "POLICE": 3}

def load_wav(fpath):
    """Load dan proses audio — sama persis dengan pipeline training."""
    audio, sr = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)
    # Pastikan panjang minimal 2 detik
    tlen = SAMPLE_RATE * 2
    if len(audio) < tlen:
        audio = np.tile(audio, int(np.ceil(tlen / len(audio))))
    audio = audio[:tlen]
    # Mel-spectrogram dengan librosa (sama dengan train_lokal.py)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # (N_MELS, T)
    mel_db = mel_db.T  # → (T, N_MELS)
    # Sesuaikan jumlah frame
    if mel_db.shape[0] < N_FRAMES:
        pad = N_FRAMES - mel_db.shape[0]
        mel_db = np.pad(mel_db, ((0, pad), (0, 0)), mode='edge')
    mel_db = mel_db[:N_FRAMES]
    # Z-score normalisasi
    mean = mel_db.mean(axis=0); std = mel_db.std(axis=0) + 1e-9
    return ((mel_db - mean) / std).astype(np.float32)

# Load model
interp = tflite.Interpreter(model_path=TFLITE_MODEL)
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
inp_scale, inp_zero = inp_det['quantization']
out_scale, out_zero = out_det['quantization']


def infer(mel2d):
    x = mel2d[np.newaxis,:,:,np.newaxis]
    if inp_det['dtype'] == np.int8:
        x = np.clip(np.round(x/inp_scale+inp_zero),-128,127).astype(np.int8)
    interp.set_tensor(inp_det['index'], x)
    interp.invoke()
    raw = interp.get_tensor(out_det['index'])[0]
    if out_det['dtype'] == np.int8:
        raw = (raw.astype(np.float32)-out_zero)*out_scale
    raw = np.maximum(raw, 0); total = raw.sum()
    return raw/total if total>0 else raw

print("="*70)
print("🔍 AUDIT FILE SALAH KAMAR — SirenMaster Dataset")
print("="*70)

salah_kamar_all = {}
total_scanned = total_salah = 0

for folder_name in ["AMBULANCE", "POLICE", "FIRETRUCK"]:
    true_cls    = FOLDER_TO_CLASS[folder_name]
    folder_path = os.path.join(ROOT, folder_name)
    wav_files   = sorted(glob.glob(os.path.join(folder_path,"*.wav")))
    print(f"\n📁 {folder_name} ({len(wav_files)} file) — sedang scan...")
    salah = []
    for i, fpath in enumerate(wav_files):
        total_scanned += 1
        try:
            pred      = infer(load_wav(fpath))
            pred_cls  = int(np.argmax(pred))
            pred_conf = float(pred[pred_cls])
            true_conf = float(pred[true_cls])
        except Exception as e:
            continue
        # Salah kamar = model prediksi kelas sirine LAIN (bukan NORMAL dan bukan benar)
        if pred_cls != true_cls and pred_cls != 2:
            salah.append({"file":fpath,"pred_cls":pred_cls,"pred_conf":pred_conf,"true_conf":true_conf,"scores":pred.tolist()})
            total_salah += 1
        if (i+1)%300==0:
            print(f"  ... {i+1}/{len(wav_files)}")
    salah_kamar_all[folder_name] = salah
    print(f"  Salah kamar: {len(salah)} file")

print(f"\n{'='*70}")
print(f"📊 TOTAL SALAH KAMAR: {total_salah} dari {total_scanned} file")
print("="*70)

if total_salah == 0:
    print("🎉 Semua file sudah di folder yang benar!")
else:
    # Tampilkan detail
    for folder_name, salah_list in salah_kamar_all.items():
        if salah_list:
            print(f"\n❌ {folder_name} — {len(salah_list)} file salah kamar:")
            for item in salah_list[:20]:
                sc = item["scores"]
                print(f"   {os.path.basename(item['file']):35s} → prediksi: {CLASS_NAMES[item['pred_cls']]:10s} ({item['pred_conf']*100:.0f}%) | AMB:{sc[0]*100:.0f}% FIRE:{sc[1]*100:.0f}% POL:{sc[3]*100:.0f}%")

    print("\nPilih tindakan:")
    print("  [1] Pindahkan ke folder yang benar (OTOMATIS)")
    print("  [2] Hapus file salah kamar")
    print("  [3] Hanya laporan, tidak ada tindakan")
    pilihan = input("\nPilihan (1/2/3): ").strip()

    if pilihan == "1":
        n = 0
        for folder_name, salah_list in salah_kamar_all.items():
            for item in salah_list:
                src      = item["file"]
                dest_dir = os.path.join(ROOT, CLASS_NAMES[item["pred_cls"]])
                dest     = os.path.join(dest_dir, os.path.basename(src))
                if os.path.exists(dest_dir):
                    os.rename(src, dest)
                    print(f"  📦 {os.path.basename(src)} → {CLASS_NAMES[item['pred_cls']]}/")
                    n += 1
        print(f"\n✅ {n} file dipindah. Silakan retrain model!")
    elif pilihan == "2":
        ok = input(f"Hapus {total_salah} file? Ketik 'HAPUS': ")
        if ok.strip().upper() == "HAPUS":
            n = 0
            for folder_name, salah_list in salah_kamar_all.items():
                for item in salah_list:
                    os.remove(item["file"]); n += 1
            print(f"✅ {n} file dihapus. Silakan retrain model!")
        else:
            print("Dibatalkan.")

print("\nSelesai.")
