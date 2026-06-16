import os
import sys
import numpy as np
import pyaudio
import tensorflow as tf
import re
import warnings
import msvcrt
import collections
import time
import scipy.io.wavfile as wav
from predict_siren_249x40 import extract_features, run_tflite_inference

# Matikan log tensorflow yang mengganggu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

SAMPLE_RATE = 8000
CHUNK_DURATION = 1.0  # Read 1 second at a time
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
BUFFER_DURATION = 4.0  # Keep 4 seconds of rolling history
BUFFER_SAMPLES = int(SAMPLE_RATE * BUFFER_DURATION)

def main():
    print("=" * 60)
    print(" 🚨 AI SIREN DETECTOR - REAL-TIME LIVE 🚨")
    print("=" * 60)
    
    model_path = "siren_model_quant.tflite"
    if not os.path.exists(model_path):
        print(f"Error: File model {model_path} tidak ditemukan!")
        sys.exit(1)
        
    model_h_path = r"siren_detection\model.h"
    if not os.path.exists(model_h_path):
        print(f"Error: model.h tidak ditemukan di {model_h_path}")
        sys.exit(1)
        
    # Membaca parameter standar dari model.h agar AI sama akuratnya dengan di ESP32
    mean, std = [], []
    with open(model_h_path, 'r') as f:
        content = f.read()
        mean_match = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\};', content)
        std_match = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\};', content)
        if mean_match and std_match:
            mean = np.array([float(x.strip().replace('f', '')) for x in mean_match.group(1).split(',')])
            std = np.array([float(x.strip().replace('f', '')) for x in std_match.group(1).split(',')])
            
    print("[*] Model dan konfigurasi berhasil dimuat.")
    print("[*] Menghidupkan Mikrofon Laptop... (Pastikan mic menyala)")
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SAMPLES)
    except Exception as e:
        print(f"\nError membuka mikrofon: {e}")
        print("Solusi: Pastikan mikrofon laptop Anda diizinkan (Privacy Settings) dan tidak digunakan aplikasi lain.")
        sys.exit(1)
        
    print("\n>>> MENDENGARKAN SUARA LINGKUNGAN / YOUTUBE ... (Tekan Ctrl+C untuk berhenti) <<<")
    print("\n[ FITUR PEREKAM CERDAS ]")
    print("Tekan [1] Simpan AMBULANCE | [2] FIRETRUCK | [3] NORMAL | [4] POLICE")
    print("Saat AI salah tebak, tekan tombol di atas untuk merekam suara ke Dataset!\n")
    
    history = collections.deque(maxlen=4) # Menyimpan 4 tebakan terakhir
    
    # State Latching (Penahan Status)
    active_alarm = None
    alarm_expire_time = 0
    
    # 4-second rolling buffer initialized with zeros
    rolling_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)
    
    try:
        while True:
            # Membaca setiap 1 detik suara dari microphone
            data = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
            new_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Geser buffer ke kiri, masukkan chunk baru di kanan
            rolling_buffer = np.roll(rolling_buffer, -CHUNK_SAMPLES)
            rolling_buffer[-CHUNK_SAMPLES:] = new_chunk
            
            # Cek volume dari 1 detik terakhir (bukan 4 detik, agar sensitif terhadap jeda)
            rms = np.sqrt(np.mean(new_chunk**2))
            if rms < 0.015:
                if time.time() < alarm_expire_time:
                    sys.stdout.write(f"\r\033[K🚨 HOLD/MENAHAN STATUS: {active_alarm} 🚨 (Jeda Suara)\n")
                else:
                    sys.stdout.write("\r[  Sunyi / Terlalu Pelan  ]                                    ")
                    sys.stdout.flush()
                history.clear()
                continue
                
            # Karena ini untuk alat pembantu Tuna Rungu (Deaf), kita TIDAK BOLEH memfilter suara.
            # Suara sirine dari luar mobil yang kedap akan terdengar "mendem" (tidak melengking).
            # Jadi kita harus mempercayakan sepenuhnya pada AI untuk mengenali polanya dari 4 DETIK penuh.
            feat = extract_features(rolling_buffer, mean, std)
            probs = run_tflite_inference(model_path, feat)
            best_idx = np.argmax(probs)
            confidence = probs[best_idx] * 100
            
            # Filter sisa: AI harus lumayan yakin (contoh: 75%+) untuk menebak
            if confidence < 75.0:
                best_idx = CATEGORIES.index('NORMAL')
            
            history.append(best_idx)
            
            if len(history) > 0:
                most_common_idx = max(set(history), key=history.count)
            else:
                most_common_idx = best_idx
                
            current_time = time.time()
            
            # Jika terdeteksi stabil (minimal 3 detik), perbarui Latch!
            if CATEGORIES[most_common_idx] != "NORMAL" and history.count(most_common_idx) >= 3:
                active_alarm = CATEGORIES[most_common_idx]
                alarm_expire_time = current_time + 4.0  # Tahan selama 4 detik ke depan
                
            # Logika Tampilan (UI)
            if current_time < alarm_expire_time:
                # Prioritaskan menampilkan alarm yang sedang di-Hold
                sys.stdout.write(f"\r\033[K🚨 TERKUNCI (LATCH): {active_alarm} 🚨\n")
            else:
                # Jika tidak ada alarm yang ditahan, tampilkan status biasa
                output = f"[  Menganalisa... (Saat ini: {CATEGORIES[best_idx]})  ]"
                sys.stdout.write(f"\r{output:<60}")
                sys.stdout.flush()
                
            # Simpan 4 detik terakhir ke file debug agar user bisa mendengar apa yang AI dengar
            # Convert float32 [-1, 1] to int16 before saving
            wav_data = (rolling_buffer * 32767).astype(np.int16)
            wav.write("debug_last_audio.wav", SAMPLE_RATE, wav_data)
            
            # Cek jika user menekan tombol 1, 2, 3, atau 4 untuk menyimpan dataset
            save_idx = -1
            while msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                if key == '1': save_idx = 0   # AMBULANCE
                elif key == '2': save_idx = 1 # FIRETRUCK
                elif key == '3': save_idx = 2 # NORMAL
                elif key == '4': save_idx = 3 # POLICE
                
            if save_idx != -1:
                cat_name = CATEGORIES[save_idx]
                save_dir = cat_name
                os.makedirs(save_dir, exist_ok=True)
                ts = int(time.time() * 1000)
                fname = os.path.join(save_dir, f"live_{cat_name.lower()}_{ts}.wav")
                wav.write(fname, SAMPLE_RATE, wav_data)
                sys.stdout.write(f"\n\033[K[+] DITAHAN: Menyimpan ke {cat_name} (4 detik)...\n")
                sys.stdout.flush()
                
    except KeyboardInterrupt:
        print("\n\nSistem dihentikan.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
