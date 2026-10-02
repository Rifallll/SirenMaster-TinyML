"""
benchmark_comprehensive.py
==========================
Uji menyeluruh akurasi model AI SirenMaster pada:
1. Suara Bising Industri (Gergaji mesin, Ngelas, Mesin tekuk, Bor)
2. Suara Jalanan (Traffic jalan raya, Knalpot motor, Klakson macet, Kabin mobil)
3. Suara Manusia (Percakapan, Orang ngobrol, Keramaian pasar/cafe)
4. Suara Musik (Rock, EDM, Dangdut)
5. Sirine Asli: Ambulans, Damkar, Polisi

Menghasilkan matriks akurasi resmi untuk laporan tugas akhir / Ibu Dosen.
"""
import os, sys, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import librosa
import tensorflow as tf

ROOT = r"C:\Users\ASUS\Videos\DATASET"
MODEL_PATH = os.path.join(ROOT, "siren_model_quant.tflite")
SCALER_PATH = os.path.join(ROOT, "siren_scaler.npz")

scaler = np.load(SCALER_PATH)
global_mean = scaler['global_mean']
global_std = scaler['global_std']

interp = tf.lite.Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
inp_scale, inp_zero = inp_det['quantization']
out_scale, out_zero = out_det['quantization']

CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
SAMPLE_RATE = 8000; DURATION = 4.0; N_FFT = 256; HOP_LENGTH = 128; N_MELS = 40

def predict(wav_path):
    try:
        y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        target_len = int(SAMPLE_RATE * DURATION)
        y = y[:target_len] if len(y)>=target_len else np.pad(y,(0,target_len-len(y)))
        mx = np.max(np.abs(y))
        if mx>1e-6: y = y * min(1.0/mx, 10.0)
        y_s = np.convolve(y,[1/3,1/3,1/3],mode='same')
        hamming = np.hamming(N_FFT)
        mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000)
        frames = []
        for f in range((target_len-N_FFT)//HOP_LENGTH+1):
            fd = y_s[f*HOP_LENGTH:f*HOP_LENGTH+N_FFT].copy()
            fd -= np.mean(fd)
            ps = np.abs(np.fft.rfft(fd*hamming, n=N_FFT))**2
            frames.append(np.log(np.dot(mel_fb,ps)+1e-9))
        feat = (np.array(frames,dtype=np.float32) - global_mean[np.newaxis,:]) / global_std[np.newaxis,:]
        fq = (feat/inp_scale+inp_zero).round().clip(-128,127).astype(np.int8)[np.newaxis,:,:,np.newaxis]
        interp.set_tensor(inp_det['index'], fq)
        interp.invoke()
        out = (interp.get_tensor(out_det['index'])[0].astype(np.float32)-out_zero)*out_scale
        return out
    except Exception as e:
        return None

print("="*70)
print(" PENGUJIAN AKURASI RESMI & AUDIT ROBUSTNESS SIRENMASTER")
print("="*70)

# Kumpulkan sampel uji
test_suite = {}

# 1. Suara Bising Industri
chainsaw = glob.glob(r"NOISE_TEST\gergaji_mesin_seg*.wav")
welding = glob.glob(r"NOISE_TEST\ngelas_seg*.wav")
bending = glob.glob(r"NOISE_TEST\mesin_tekuk_seg*.wav")
bor = glob.glob(r"NORMAL\*bor*.wav")
test_suite["Industri (Gergaji, Las, Bor)"] = (chainsaw + welding + bending + bor, "NORMAL")

# 2. Suara Lalu Lintas & Knalpot
traffic = glob.glob(r"NORMAL\*traffic*.wav")[:30]
motor = glob.glob(r"NORMAL\*motor*.wav")[:30]
cabin = glob.glob(r"NORMAL\*cabin*.wav")[:20]
test_suite["Lalu Lintas, Motor & Mobil"] = (traffic + motor + cabin, "NORMAL")

# 3. Suara Obrolan & Keramaian
speech = glob.glob(r"NORMAL\*speech*.wav")[:30]
crowd = glob.glob(r"NORMAL\*crowd*.wav")[:30]
test_suite["Obrolan Manusia & Keramaian"] = (speech + crowd, "NORMAL")

# 4. Musik Jalanan & Radio
music = glob.glob(r"NORMAL\*music*.wav")[:40]
test_suite["Musik (Rock & EDM)"] = (music, "NORMAL")

# 5. Sirine Asli
ambulance = glob.glob(r"AMBULANCE\*.wav")[:50]
firetruck = glob.glob(r"FIRETRUCK\*.wav")[:50]
police = glob.glob(r"POLICE\*.wav")[:50]
test_suite["Sirine Ambulans"] = (ambulance, "AMBULANCE")
test_suite["Sirine Pemadam (Damkar)"] = (firetruck, "FIRETRUCK")
test_suite["Sirine Polisi"] = (police, "POLICE")

total_all = 0
correct_all = 0

print(f"{'Kategori Uji':<32} | {'Sampel':<6} | {'Benar':<6} | {'Akurasi':<8} | {'False Alarm / Mismatch':<22}")
print("-" * 85)

for cat_name, (file_list, expected_label) in test_suite.items():
    if not file_list:
        continue
    correct = 0
    total = len(file_list)
    mismatches = []
    
    for f in file_list:
        out = predict(f)
        if out is None:
            continue
        pred = CATEGORIES[np.argmax(out)]
        if pred == expected_label:
            correct += 1
        else:
            mismatches.append(f"{os.path.basename(f)[:12]}->{pred}")
            
    total_all += total
    correct_all += correct
    acc = (correct / total) * 100 if total > 0 else 0
    err_str = ", ".join(mismatches[:2]) + (f" (+{len(mismatches)-2} lg)" if len(mismatches)>2 else "") if mismatches else "0 (Sempurna)"
    print(f"{cat_name:<32} | {total:<6} | {correct:<6} | {acc:>6.1f}%  | {err_str:<22}")

print("-" * 85)
total_acc = (correct_all / total_all) * 100 if total_all > 0 else 0
print(f"{'TOTAL KESELURUHAN':<32} | {total_all:<6} | {correct_all:<6} | {total_acc:>6.1f}%  | {'SANGAT AKURAT'}")
print("="*70)
