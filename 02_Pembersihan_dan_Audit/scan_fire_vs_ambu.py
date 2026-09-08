"""
SCAN KHUSUS: Cari file FIRETRUCK yang masih salah dikira AMBULANCE
Langsung dari dataset asli, bukan dummy
"""
import os, sys, re, numpy as np, librosa
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE         = r"C:\Users\ASUS\Videos\DATASET"
MODEL_PATH   = os.path.join(BASE, "siren_model_quant.tflite")
CATEGORIES   = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
SR           = 8000
N_FFT        = 256
HOP          = 128
N_MELS       = 40
DUR          = 4.0
TARGET       = int(SR * DUR)

# Load normalization
MODEL_H = os.path.join(BASE, "sirenmaster_main", "model.h")
with open(MODEL_H, 'r') as f:
    c = f.read()
mm = re.search(r'MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\}', c)
ms = re.search(r'MEL_STD\[\d+\] PROGMEM = \{([^}]+)\}', c)
MEL_MEAN = np.array([float(x.strip().rstrip('f')) for x in mm.group(1).split(',')])
MEL_STD  = np.array([float(x.strip().rstrip('f')) for x in ms.group(1).split(',')])

_ham = np.hamming(N_FFT).astype(np.float32)
_mel = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000)

def feat(y):
    if len(y) < TARGET: y = np.pad(y, (0, TARGET - len(y)))
    else: y = y[:TARGET]
    y = np.convolve(y, [1/3,1/3,1/3], mode='same')
    frames = []
    for i in range((TARGET - N_FFT) // HOP + 1):
        f = y[i*HOP:i*HOP+N_FFT].copy()
        f -= f.mean(); f *= _ham
        p = np.abs(np.fft.rfft(f, N_FFT))**2
        frames.append(np.log(_mel @ p[:N_FFT//2+1] + 1e-9))
    spec = np.array(frames, np.float32)
    return (spec - MEL_MEAN) / (MEL_STD + 1e-8)

interp = tf.lite.Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

def predict(spec):
    x = spec[np.newaxis,:,:,np.newaxis].astype(np.float32)
    sc = inp['quantization_parameters']['scales'][0]
    zp = inp['quantization_parameters']['zero_points'][0]
    xq = np.clip(np.round(x/sc)+zp, -128, 127).astype(np.int8)
    interp.set_tensor(inp['index'], xq)
    interp.invoke()
    o = interp.get_tensor(out['index'])[0].astype(np.float32)
    sc2 = out['quantization_parameters']['scales'][0]
    zp2 = out['quantization_parameters']['zero_points'][0]
    return (o - zp2) * sc2

print("=" * 65)
print("  SCAN FIRETRUCK → yang masih dikira AMBULANCE")
print("  Menggunakan model terbaru + dataset asli")
print("=" * 65)

folder = os.path.join(BASE, "FIRETRUCK")
files  = sorted([f for f in os.listdir(folder)
                 if f.lower().endswith(('.wav','.m4a'))])

print(f"\n  Scanning {len(files)} file FIRETRUCK...\n")

salah_ambu  = []
salah_lain  = []
benar       = 0

for fname in files:
    try:
        y, _ = librosa.load(os.path.join(folder, fname),
                            sr=SR, mono=True, duration=DUR)
        probs   = predict(feat(y))
        pred    = int(np.argmax(probs))
        conf    = float(probs[pred])
        if pred == 1:        # benar FIRETRUCK
            benar += 1
        elif pred == 0:      # dikira AMBULANCE
            salah_ambu.append((fname, conf, probs))
        else:
            salah_lain.append((fname, CATEGORIES[pred], conf))
    except:
        pass

# Tampilkan yang salah dikira AMBULANCE
salah_ambu.sort(key=lambda x: -x[1])

print(f"  Hasil:")
print(f"  ✅ Benar (FIRETRUCK)   : {benar}")
print(f"  ❌ Salah → AMBULANCE  : {len(salah_ambu)}")
print(f"  ❌ Salah → lainnya    : {len(salah_lain)}")

if salah_ambu:
    print(f"\n{'='*65}")
    print(f"  FILE FIRETRUCK YANG MASIH DIKIRA AMBULANCE:")
    print(f"{'='*65}")
    for fname, conf, probs in salah_ambu:
        a,fi,n,p = probs
        print(f"\n  ❌ {fname}")
        print(f"     Yakin AMBULANCE: {conf*100:.1f}%")
        print(f"     [AMBU:{a*100:.0f}% FIRE:{fi*100:.0f}% NORM:{n*100:.0f}% POLI:{p*100:.0f}%]")

    # Kelompokkan berdasarkan sumber rekaman
    print(f"\n{'='*65}")
    print(f"  GRUP SUMBER REKAMAN BERMASALAH:")
    print(f"{'='*65}")
    from collections import defaultdict
    grp = defaultdict(list)
    for fname, conf, _ in salah_ambu:
        base = re.sub(r'^aug_\w+_\d+_', '', os.path.splitext(fname)[0])
        base = re.sub(r'_(loud|original|noise_light|shift|seg\d+)$', '', base)
        base = re.sub(r'_\d+$', '', base)
        grp[base].append((fname, conf))
    for src, items in sorted(grp.items(), key=lambda x: -len(x[1])):
        avg = sum(c for _,c in items)/len(items)
        print(f"\n  Grup '{src}' → {len(items)} file (rata-rata yakin AMBULANCE: {avg*100:.0f}%)")
        for f,c in items[:3]:
            print(f"    - {f}  ({c*100:.1f}%)")
        if len(items) > 3:
            print(f"    ... dan {len(items)-3} lainnya")
