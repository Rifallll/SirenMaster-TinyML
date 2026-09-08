import numpy as np, librosa, os, sys, glob, re, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SAMPLE_RATE = 8000
DURATION = 4.0
TARGET_LEN = int(SAMPLE_RATE * DURATION)
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

MODEL_H = r'sirenmaster_main\model.h'
with open(MODEL_H, 'r') as f:
    content = f.read()
mean_match = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\}', content)
std_match  = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\}', content)
MEL_MEAN = np.array([float(x.strip().rstrip('f')) for x in mean_match.group(1).split(',')])
MEL_STD  = np.array([float(x.strip().rstrip('f')) for x in std_match.group(1).split(',')])

def extract_features(y):
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    else:
        y = y[:TARGET_LEN]
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y * min(1.0 / max_val, 10.0)
    y = np.convolve(y, [1/3, 1/3, 1/3], mode='same')
    n_frames = (TARGET_LEN - 256) // 128 + 1
    hamming = np.hamming(256)
    mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=256, n_mels=40, fmin=0, fmax=4000)
    log_mel_frames = []
    for i in range(n_frames):
        start = i * 128
        frame = y[start:start+256].copy()
        frame -= np.mean(frame)
        frame *= hamming
        fft = np.fft.rfft(frame, n=256)
        power = np.abs(fft)**2
        mel = mel_fb @ power[:129]
        log_mel_frames.append(np.log(mel + 1e-9))
    spec = np.array(log_mel_frames, dtype=np.float32)
    spec = (spec - MEL_MEAN) / (MEL_STD + 1e-8)
    return spec

interpreter = tf.lite.Interpreter(model_path='siren_model_quant.tflite')
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

def predict(spec):
    inp = spec[np.newaxis, :, :, np.newaxis].astype(np.float32)
    if inp_det['dtype'] == np.int8:
        sc = inp_det['quantization_parameters']['scales'][0]
        zp = inp_det['quantization_parameters']['zero_points'][0]
        inp_q = np.clip(np.round(inp / sc) + zp, -128, 127).astype(np.int8)
        interpreter.set_tensor(inp_det['index'], inp_q)
        interpreter.invoke()
        out = interpreter.get_tensor(out_det['index'])[0].astype(np.float32)
        sc2 = out_det['quantization_parameters']['scales'][0]
        zp2 = out_det['quantization_parameters']['zero_points'][0]
        return (out - zp2) * sc2
    else:
        interpreter.set_tensor(inp_det['index'], inp)
        interpreter.invoke()
        return interpreter.get_tensor(out_det['index'])[0]

# Full audit per class
print("=" * 70)
print("  AUDIT LINTAS KELAS - Mencari file yang misprediksi")
print("=" * 70)

all_stats = {}
for cat_idx, cat in enumerate(CATEGORIES):
    files = glob.glob(f'{cat}/*.wav')
    correct = 0
    wrong = []
    for fp in files:
        try:
            y, _ = librosa.load(fp, sr=SAMPLE_RATE)
            spec = extract_features(y)
            probs = predict(spec)
            pred = np.argmax(probs)
            if pred == cat_idx:
                correct += 1
            else:
                wrong.append({
                    'file': os.path.basename(fp),
                    'pred': CATEGORIES[pred],
                    'pred_score': float(probs[pred]),
                    'true_score': float(probs[cat_idx])
                })
        except Exception as e:
            pass

    pct = correct / len(files) * 100 if files else 0
    print(f"\n{cat}: {correct}/{len(files)} correct ({pct:.1f}%)")
    print(f"  Mispredicted: {len(wrong)} files")

    # Show top worst mismatches
    wrong_sorted = sorted(wrong, key=lambda x: x['pred_score'], reverse=True)
    if wrong_sorted:
        print(f"  Top mispredictions (high confidence wrongs):")
        for w in wrong_sorted[:5]:
            print(f"    {w['file'][:45]:45s} -> {w['pred']} ({w['pred_score']*100:.0f}%) true_score={w['true_score']*100:.0f}%")

    all_stats[cat] = {'total': len(files), 'correct': correct, 'wrong': wrong}

# Save suspicious police files to a list
suspicious_police = [w['file'] for w in all_stats.get('POLICE', {}).get('wrong', []) if w['pred'] == 'AMBULANCE' and w['pred_score'] > 0.80]
print(f"\n\n=== POLICE files predicted as AMBULANCE >80% confidence ===")
print(f"Total: {len(suspicious_police)} files")
for fn in suspicious_police:
    print(f"  POLICE/{fn}")

with open('audit_police_ambiguous_result.txt', 'w', encoding='utf-8') as f:
    f.write("POLICE files strongly predicted as AMBULANCE (candidates to move/delete):\n")
    for fn in suspicious_police:
        f.write(f"POLICE/{fn}\n")
print(f"\nList saved to audit_police_ambiguous_result.txt")
