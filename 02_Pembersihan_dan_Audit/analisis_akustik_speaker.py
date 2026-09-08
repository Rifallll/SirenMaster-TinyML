"""
analisis_akustik_speaker.py
===========================
Simulasi efek speaker laptop terhadap prediksi model:
Bandingkan skor model untuk audio ASLI (WAV) vs 
audio yang disimulasikan melewati karakteristik speaker laptop.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, glob, numpy as np, librosa, tensorflow as tf

ROOT   = r"C:\Users\ASUS\Videos\DATASET"
TFLITE = os.path.join(ROOT, "sirenmaster_main", "model.tflite")
SCALER = os.path.join(ROOT, "siren_scaler.npz")

scaler = np.load(SCALER)
g_mean = scaler['global_mean']
g_std  = scaler['global_std']

interp = tf.lite.Interpreter(model_path=TFLITE)
interp.allocate_tensors()
in_d  = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
in_s, in_z   = in_d['quantization']
out_s, out_z = out_d['quantization']

CATS = ['AMBULANCE', 'FIRETRUCK', 'NOISE', 'POLICE']
ham    = np.hamming(256)
mel_fb = librosa.filters.mel(sr=8000, n_fft=256, n_mels=40, fmin=0, fmax=4000)

def simulate_laptop_speaker(y, sr=8000):
    """Simulasikan karakteristik speaker laptop:
    - High-pass filter (laptop speaker mulai lemah di bawah 200Hz)
    - Room reverb ringan
    - Normalisasi ulang setelah distorsi
    """
    from scipy.signal import butter, lfilter, sosfilt
    
    # 1. High-pass filter 200Hz (laptop speaker tidak bisa reproduce bass)
    sos_hp = butter(4, 200/(sr/2), btype='high', output='sos')
    y_hp = sosfilt(sos_hp, y)
    
    # 2. Mid-range boost 800-3000Hz (speaker laptop emphasize mid range)
    sos_bp = butter(2, [800/(sr/2), 3000/(sr/2)], btype='band', output='sos')
    y_mid = sosfilt(sos_bp, y)
    y_combined = y_hp + 0.5 * y_mid
    
    # 3. Simulasi reverb ringan (delay ~50ms, attenuation 0.3)
    delay_samples = int(0.05 * sr)  # 50ms
    y_reverb = np.zeros_like(y_combined)
    y_reverb[:len(y_combined)] = y_combined
    if len(y_combined) > delay_samples:
        y_reverb[delay_samples:] += 0.3 * y_combined[:len(y_combined)-delay_samples]
    
    # 4. Normalisasi
    peak = np.max(np.abs(y_reverb))
    if peak > 0:
        y_reverb = y_reverb / peak * 0.8
    
    return y_reverb

def predict(y):
    dc = np.mean(y); y = y - dc
    nf = (len(y) - 256) // 128 + 1
    frames = []
    for f in range(nf):
        fd = y[f*128:f*128+256]
        if len(fd) < 256: fd = np.pad(fd, (0, 256-len(fd)))
        mel_e = np.dot(mel_fb, np.abs(np.fft.rfft(fd*ham, n=256))**2)
        frames.append(np.log(mel_e + 1e-9))
    feat = (np.array(frames, np.float32) - g_mean) / g_std
    feat = feat.reshape(in_d['shape'])
    inp = np.round(feat/in_s+in_z).astype(np.int8)
    interp.set_tensor(in_d['index'], inp)
    interp.invoke()
    raw = interp.get_tensor(out_d['index'])[0]
    scores = (raw.astype(np.float32)-out_z)*out_s
    s = np.maximum(0, scores); s /= (s.sum()+1e-9)
    return s

test_files = [
    ("AMBULANCE", os.path.join(ROOT, "AMBULANCE", "ambulance_0004_seg01.wav"), 0),
    ("FIRETRUCK", os.path.join(ROOT, "FIRETRUCK", "fire_0002_seg01.wav"),      1),
    ("POLICE",    os.path.join(ROOT, "POLICE",    "police_0001_seg04.wav"),     3),
    ("POLICE",    os.path.join(ROOT, "POLICE",    "police_0001_seg05.wav"),     3),
    ("POLICE",    os.path.join(ROOT, "POLICE",    "police_0001_seg03.wav"),     3),
]

print("=" * 90)
print(" PERBANDINGAN SKOR: AUDIO ASLI vs SIMULASI SPEAKER LAPTOP")
print("=" * 90)
print(f"{'Kelas':<12} | {'File':<30} | {'Kondisi':<15} | A%   F%   N%   P%   | Prediksi")
print("-" * 90)

for class_name, fp, expected in test_files:
    fn = os.path.basename(fp)
    y, _ = librosa.load(fp, sr=8000)
    
    # Asli (digital)
    s1 = predict(y.copy())
    pred1 = CATS[np.argmax(s1)]
    ok1 = (np.argmax(s1) == expected)
    status1 = "✅" if ok1 else "❌"
    print(f"{class_name:<12} | {fn:<30} | {'WAV ASLI':<15} | {s1[0]*100:3.0f}  {s1[1]*100:3.0f}  {s1[2]*100:3.0f}  {s1[3]*100:3.0f}  | {pred1} {status1}")
    
    # Simulasi speaker laptop
    y_spk = simulate_laptop_speaker(y.copy())
    s2 = predict(y_spk)
    pred2 = CATS[np.argmax(s2)]
    ok2 = (np.argmax(s2) == expected)
    status2 = "✅" if ok2 else "❌"
    print(f"{class_name:<12} | {fn:<30} | {'VIA SPEAKER':<15} | {s2[0]*100:3.0f}  {s2[1]*100:3.0f}  {s2[2]*100:3.0f}  {s2[3]*100:3.0f}  | {pred2} {status2}")
    print("-" * 90)

print()
print("KETERANGAN:")
print("  WAV ASLI = Prediksi model dari file digital langsung (seperti Python audit)")
print("  VIA SPEAKER = Simulasi audio melewati speaker laptop + reverb ruangan")
print("  Perbedaan ini adalah penyebab ESP32 selalu menebak Ambulance!")
print()

# Tambahan: Scan semua file POLICE untuk cari yang paling robust
print("=" * 90)
print(" SCAN SEMUA FILE POLICE: Cari yang paling robust via speaker")
print("=" * 90)
police_files = sorted(glob.glob(os.path.join(ROOT, "POLICE", "*.wav")))[:20]
results = []
for fp in police_files:
    fn = os.path.basename(fp)
    y, _ = librosa.load(fp, sr=8000)
    y_spk = simulate_laptop_speaker(y.copy())
    s = predict(y_spk)
    pol_score = s[3]
    amb_score = s[0]
    results.append((pol_score, amb_score, fn))

results.sort(reverse=True)
print(f"{'File':<35} | Pol%  | Amb%  | Status")
print("-" * 70)
for pol, amb, fn in results[:10]:
    status = "ROBUST" if pol > amb else "GAGAL"
    print(f"{fn:<35} | {pol*100:4.0f}  | {amb*100:4.0f}  | {status}")
