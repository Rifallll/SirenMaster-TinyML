"""
audit_lcd_display_and_inference.py
====================================
Simulasi Real-Time 100% C++ ESP32 (Frame per 100ms selama 5 detik)
untuk Mengaudit Respon Klasifikasi AI & Tampilan Display LCD saat Buffer Penuh.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os, glob, librosa, numpy as np
import tensorflow as tf

ROOT = r"C:\Users\ASUS\Videos\DATASET"
TFLITE = os.path.join(ROOT, "sirenmaster_main", "model.tflite")
if not os.path.exists(TFLITE):
    TFLITE = os.path.join(ROOT, "04_Training_AI", "siren_model_quant.tflite")

SCALER_ESP32 = os.path.join(ROOT, "siren_scaler.npz")
scaler_data = np.load(SCALER_ESP32)
g_mean, g_std = scaler_data['global_mean'], scaler_data['global_std']

CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NOISE', 'POLICE']
LCD_LABELS = ['AMBULANCE', 'DAMKAR',    'AMAN',  'POLISI']

interp = tf.lite.Interpreter(model_path=TFLITE)
interp.allocate_tensors()
in_idx = interp.get_input_details()[0]['index']
out_idx = interp.get_output_details()[0]['index']
in_s, in_z = interp.get_input_details()[0]['quantization']
out_s, out_z = interp.get_output_details()[0]['quantization']
is_quant = (in_s != 0.0)
shape = interp.get_input_details()[0]['shape']

CONFIDENCE_THR = 0.50
EMA_ALPHA = 0.15

class ESP32Simulator:
    def __init__(self, mic_gain=1.0):
        self.mic_gain = mic_gain
        self.ema_probs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self.alert_lock_time = 0
        self.locked_alert_class = 2
        self.locked_alert_conf = 0.0
        self.weak_siren_streak = 0
        self.last_weak_siren_class = 2
        self.time_ms = 0

    def simulate_frame(self, audio_chunk_32k):
        dc = np.mean(audio_chunk_32k)
        clean_audio = (audio_chunk_32k - dc) * self.mic_gain
        clean_audio = np.clip(clean_audio, -1.0, 1.0)
        rms = np.sqrt(np.mean(clean_audio**2)) * 32768.0

        n_frames = (len(clean_audio) - 256) // 128 + 1
        hamming = np.hamming(256)
        mel_fb = librosa.filters.mel(sr=8000, n_fft=256, n_mels=40, fmin=0, fmax=4000)

        log_mel_frames = []
        for frame in range(n_frames):
            start = frame * 128
            fd = clean_audio[start:start+256].copy()
            fft_out = np.fft.rfft(fd * hamming, n=256)
            power = np.abs(fft_out) ** 2
            mel_e = np.dot(mel_fb, power)
            log_mel_frames.append(np.log(mel_e + 1e-9))

        feat = np.array(log_mel_frames, dtype=np.float32)
        feat = (feat - g_mean) / g_std
        feat = np.reshape(feat, shape)

        if is_quant:
            input_data = np.round(feat / in_s + in_z).astype(np.int8)
        else:
            input_data = feat
        interp.set_tensor(in_idx, input_data)
        interp.invoke()
        out = interp.get_tensor(out_idx)[0]
        if is_quant:
            scores = (out.astype(np.float32) - out_z) * out_s
        else:
            scores = out

        cs = np.maximum(0.0, scores)
        sum_cs = np.sum(cs)
        if sum_cs > 0: cs /= sum_cs
        else: cs = np.array([0.0, 0.0, 1.0, 0.0])

        rawBest = int(np.argmax(cs))
        rawScore = float(cs[rawBest])

        for i in range(4):
            self.ema_probs[i] = (1.0 - EMA_ALPHA) * self.ema_probs[i] + EMA_ALPHA * cs[i]
        es = np.sum(self.ema_probs)
        if es > 0: self.ema_probs /= es

        best = int(np.argmax(self.ema_probs))
        bp = float(self.ema_probs[best])

        thinking = False
        out_conf = 0.0
        winClass = 2

        # 1. Cek streak
        if rawBest != 2 and rawBest == self.last_weak_siren_class:
            self.weak_siren_streak += 1
        elif rawBest != 2:
            self.weak_siren_streak = 1
            self.last_weak_siren_class = rawBest
        else:
            self.weak_siren_streak = 0
            self.last_weak_siren_class = 2

        # 2. Cek Lock 3 detik
        if self.alert_lock_time > 0 and self.time_ms - self.alert_lock_time < 3000:
            if rawBest != 2 and rawBest != self.locked_alert_class and self.weak_siren_streak >= 3 and rawScore >= 0.80:
                self.locked_alert_class = rawBest
                self.locked_alert_conf = rawScore
                self.alert_lock_time = self.time_ms
                out_conf = rawScore
                winClass = self.locked_alert_class
            else:
                if rawBest != 2 and rawBest == self.locked_alert_class and rawScore >= 0.78:
                    self.locked_alert_conf = max(self.locked_alert_conf, rawScore)
                    self.alert_lock_time = self.time_ms
                out_conf = self.locked_alert_conf
                winClass = self.locked_alert_class
        else:
            # 3. Deteksi Cepat
            if rawBest != 2 and self.weak_siren_streak >= 2 and (rawScore >= 0.78 or (best != 2 and bp >= 0.75)):
                winClass = rawBest if (rawScore >= 0.78) else best
                winScore = max(rawScore, bp)
                self.alert_lock_time = self.time_ms
                self.locked_alert_class = winClass
                self.locked_alert_conf = winScore
                out_conf = winScore
            else:
                winClass = 2
                out_conf = bp

        lcd_text = LCD_LABELS[winClass]
        if thinking: lcd_text = "DETEKSI"

        return winClass, out_conf, scores, rms, lcd_text, self.weak_siren_streak, rawBest, rawScore

print("=" * 95)
print(" 🖥️ AUDIT SIMULASI C++ ESP32 & TAMPILAN DISPLAY LCD UNTUK SUARA DAMKAR (REAL-TIME)")
print("=" * 95)

test_files = [
    os.path.join(ROOT, "FIRETRUCK", "fire_0002_seg01.wav"),
    os.path.join(ROOT, "FIRETRUCK", "fire_0002_seg03.wav"),
]

for fp in test_files:
    if not os.path.exists(fp): continue
    fn = os.path.basename(fp)
    print(f"\n[+] Memutar Suara DAMKAR : {fn} (Simulasi 0 s.d 4.5 Detik)")
    print("-" * 95)
    print(f"{'Waktu (ms)':<10} | {'Raw AI (Amb/Fire/Norm/Pol)':<34} | {'Streak':<7} | {'Raw Best':<14} | {'Tampilan LCD ESP32':<18}")
    print("-" * 95)
    
    sim = ESP32Simulator(mic_gain=1.0)
    y, _ = librosa.load(fp, sr=8000)
    # Ulang audio dua kali agar cukup panjang 8 detik
    y = np.tile(y, 2)
    
    # Simulasi historyBuffer 32,000 (4 detik)
    history_buf = np.zeros(32000, dtype=np.float32)
    
    for step in range(45): # 45 langkah * 100ms = 4.5 detik
        sim.time_ms = step * 100
        audio_chunk = y[step*800 : (step+1)*800]
        if len(audio_chunk) > 0:
            history_buf[:-len(audio_chunk)] = history_buf[len(audio_chunk):]
            history_buf[-len(audio_chunk):] = audio_chunk
            
        winClass, out_conf, scores, rms, lcd_text, streak, rawBest, rawScore = sim.simulate_frame(history_buf)
        
        # Cetak hanya mulai detik ke-2 saat audio sudah mulai mendominasi ring buffer
        if step >= 15:
            scores_str = f"A:{scores[0]*100:.0f}% F:{scores[1]*100:.0f}% N:{scores[2]*100:.0f}% P:{scores[3]*100:.0f}%"
            raw_str = f"{CATEGORIES[rawBest]} ({rawScore*100:.0f}%)"
            print(f"{sim.time_ms:<10} | {scores_str:<34} | {streak:<7} | {raw_str:<14} | [{lcd_text:^16}]")
