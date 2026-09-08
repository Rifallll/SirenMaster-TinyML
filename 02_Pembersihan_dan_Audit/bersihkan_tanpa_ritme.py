"""
bersihkan_tanpa_ritme.py
========================
Hapus HANYA file yang:
  1. Model salah kamar dengan confidence > 90% (sangat yakin bukan kelasnya)
  2. BUKAN file SYNTH_* (SYNTH terbukti dari spektrogram bentuknya benar)
  3. BUKAN file yang ambiguous (confidence < 90% = mungkin masih oke)

Tampilkan laporan DULU sebelum ada penghapusan.
"""
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

SAMPLE_RATE  = 8000
N_FFT        = 256
HOP_LEN      = 128
N_MELS       = 40
N_FRAMES     = 249

CLASS_NAMES     = {0: "AMBULANCE", 1: "FIRETRUCK", 2: "NORMAL", 3: "POLICE"}
FOLDER_TO_CLASS = {"AMBULANCE": 0, "FIRETRUCK": 1, "POLICE": 3}

# ── ATURAN KEAMANAN ────────────────────────────────────────────────────────
# Threshold: hanya hapus jika model yakin > X% file ini bukan kelasnya
CONFIDENCE_THR = 0.90   # 90% — sangat yakin salah

# Prefix file yang TIDAK BOLEH dihapus dalam kondisi apapun
PROTECTED_PREFIXES = ["SYNTH_"]   # File buatan sendiri, terbukti benar di spektrogram

def is_protected(fname):
    for pref in PROTECTED_PREFIXES:
        if os.path.basename(fname).startswith(pref):
            return True
    return False

def load_wav(fpath):
    audio, _ = librosa.load(fpath, sr=SAMPLE_RATE, mono=True, duration=3.0)
    tlen = SAMPLE_RATE * 2
    if len(audio) < tlen:
        audio = np.tile(audio, int(np.ceil(tlen / len(audio))))
    audio = audio[:tlen]
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).T
    if mel_db.shape[0] < N_FRAMES:
        mel_db = np.pad(mel_db, ((0, N_FRAMES - mel_db.shape[0]), (0, 0)), mode='edge')
    mel_db = mel_db[:N_FRAMES]
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
    x = mel2d[np.newaxis, :, :, np.newaxis]
    if inp_det['dtype'] == np.int8:
        x = np.clip(np.round(x / inp_scale + inp_zero), -128, 127).astype(np.int8)
    interp.set_tensor(inp_det['index'], x)
    interp.invoke()
    raw = interp.get_tensor(out_det['index'])[0]
    if out_det['dtype'] == np.int8:
        raw = (raw.astype(np.float32) - out_zero) * out_scale
    raw = np.maximum(raw, 0)
    total = raw.sum()
    return raw / total if total > 0 else raw

print("=" * 70)
print("🔍 AUDIT KETAT: Cari file tanpa ritme sirine (confidence >90%)")
print(f"   Threshold: >{CONFIDENCE_THR*100:.0f}% confidence salah")
print(f"   Dilindungi: {PROTECTED_PREFIXES}")
print("=" * 70)

kandidat_hapus = []   # (filepath, true_folder, pred_cls, pred_conf, scores)
total_scanned  = 0
total_dilindungi = 0
total_ambiguous  = 0

for folder_name in ["AMBULANCE", "POLICE", "FIRETRUCK"]:
    true_cls    = FOLDER_TO_CLASS[folder_name]
    folder_path = os.path.join(ROOT, folder_name)
    wav_files   = sorted(glob.glob(os.path.join(folder_path, "*.wav")))

    print(f"\n📁 {folder_name} ({len(wav_files)} file)...")
    folder_kandidat = 0

    for i, fpath in enumerate(wav_files):
        total_scanned += 1

        # Cek proteksi dulu
        if is_protected(fpath):
            total_dilindungi += 1
            continue

        try:
            pred      = infer(load_wav(fpath))
            pred_cls  = int(np.argmax(pred))
            pred_conf = float(pred[pred_cls])
        except Exception:
            continue

        # Salah kamar DAN confidence sangat tinggi (>90%)
        if pred_cls != true_cls and pred_cls != 2 and pred_conf >= CONFIDENCE_THR:
            kandidat_hapus.append((fpath, folder_name, pred_cls, pred_conf, pred.tolist()))
            folder_kandidat += 1
        elif pred_cls != true_cls and pred_cls != 2:
            total_ambiguous += 1   # Salah kamar tapi confidence < 90%, jangan hapus

        if (i+1) % 300 == 0:
            print(f"  ... {i+1}/{len(wav_files)}")

    print(f"  Kandidat hapus (>90%): {folder_kandidat} file")

# ── Laporan ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"📊 RINGKASAN AUDIT KETAT")
print(f"   Total scan    : {total_scanned} file")
print(f"   Dilindungi    : {total_dilindungi} file (SYNTH_* — tidak disentuh)")
print(f"   Ambiguous     : {total_ambiguous} file (conf <90% — tidak disentuh)")
print(f"   Kandidat hapus: {len(kandidat_hapus)} file (>90% yakin bukan kelasnya)")
print("=" * 70)

if len(kandidat_hapus) == 0:
    print("\n🎉 Tidak ada file yang perlu dihapus! Dataset sudah bersih.")
else:
    # Kelompokkan per folder
    per_folder = {"AMBULANCE": [], "POLICE": [], "FIRETRUCK": []}
    for item in kandidat_hapus:
        per_folder[item[1]].append(item)

    for folder_name, items in per_folder.items():
        if items:
            print(f"\n❌ {folder_name} — {len(items)} file kandidat hapus:")
            for fpath, fld, pred_cls, pred_conf, scores in items[:15]:
                sc = scores
                print(f"   {os.path.basename(fpath):40s} "
                      f"→ {CLASS_NAMES[pred_cls]:10s} ({pred_conf*100:.0f}%) "
                      f"| AMB:{sc[0]*100:.0f}% FIRE:{sc[1]*100:.0f}% POL:{sc[3]*100:.0f}%")
            if len(items) > 15:
                print(f"   ... dan {len(items)-15} file lainnya")

    print(f"\n{'='*70}")
    print(f"PERHATIAN: {len(kandidat_hapus)} file akan dihapus permanen!")
    print(f"File SYNTH_* ({total_dilindungi} file) TIDAK akan tersentuh.")
    print(f"File ambiguous ({total_ambiguous} file) TIDAK akan tersentuh.")
    print(f"{'='*70}")
    print("\nPilihan:")
    print("  [1] HAPUS semua kandidat di atas (tidak bisa dibatalkan!)")
    print("  [2] BATAL — tidak hapus apa-apa")

    pilihan = input("\nPilihan Anda (1/2): ").strip()

    if pilihan == "1":
        konfirm = input(f"Konfirmasi: Ketik 'HAPUS {len(kandidat_hapus)}' untuk lanjutkan: ").strip()
        expected = f"HAPUS {len(kandidat_hapus)}"
        if konfirm == expected:
            n = 0
            for fpath, fld, pred_cls, pred_conf, scores in kandidat_hapus:
                try:
                    os.remove(fpath)
                    n += 1
                except Exception as e:
                    print(f"  [ERR] {os.path.basename(fpath)}: {e}")
            print(f"\n✅ {n} file berhasil dihapus!")
            print("   Silakan retrain model setelah ini.")
        else:
            print(f"Konfirmasi salah (harus: '{expected}'). Dibatalkan.")
    else:
        print("Dibatalkan. Tidak ada file yang dihapus.")

print("\nSelesai.")
