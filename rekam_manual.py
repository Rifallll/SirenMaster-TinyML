import os, sys, time, msvcrt
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: pip install sounddevice")
    sys.exit(1)
    
try:
    import scipy.io.wavfile as wav
except ImportError:
    print("ERROR: pip install scipy")
    sys.exit(1)

CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
SAMPLE_RATE = 8000

# Pastikan foldernya ada
for cat in CATEGORIES:
    os.makedirs(cat, exist_ok=True)

print("="*60)
print(" 🎙️ ALAT PEREKAM DATASET ANTI-PUTUS (GAPLESS) 🎙️")
print("="*60)
print("Cara Pakai:")
print(" - Tekan '1' untuk MULAI Merekam AMBULANS")
print(" - Tekan '2' untuk MULAI Merekam DAMKAR")
print(" - Tekan '4' untuk MULAI Merekam POLISI")
print(" - Tekan 'SPASI' (atau tombol apa saja) untuk BERHENTI & SIMPAN")
print(" - Tekan Ctrl+C untuk keluar jika sudah selesai.")
print("="*60)

is_recording = False
current_cat_idx = -1
recorded_audio = []
latest_rms = 0.0

# Callback ini berjalan di latar belakang (Background Thread) oleh sistem operasi
# Fungsinya merekam suara terus menerus 100% tanpa ada celah (gapless)
def audio_callback(indata, frames, time_info, status):
    global is_recording, recorded_audio, latest_rms
    # indata shape adalah (frames, channels)
    audio_chunk = indata[:, 0].copy()
    latest_rms = np.sqrt(np.mean(audio_chunk ** 2))
    
    if is_recording:
        recorded_audio.append(audio_chunk)

try:
    # Membuka jalur rekaman non-stop ke mikrofon
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=int(SAMPLE_RATE*0.1), callback=audio_callback)
    with stream:
        while True:
            # Layar di-update setiap 0.1 detik, TANPA menghentikan rekaman mic
            if is_recording:
                # Hitung durasi rekaman
                total_samples = sum(len(c) for c in recorded_audio)
                sec = total_samples / SAMPLE_RATE
                print(f" [MEREKAM] 🔴 {CATEGORIES[current_cat_idx]} ... ({sec:.1f} detik) (Tekan SPASI untuk Stop)   ", end='\r')
            else:
                if latest_rms < 0.0005:
                    print(f" [Siap] Hening... (Tekan 1/2/4 untuk merekam)            ", end='\r')
                else:
                    print(f" [Siap] Suara Terdengar! (Tekan 1/2/4 untuk merekam)     ", end='\r')
            
            # Cek keyboard
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                
                if not is_recording:
                    if key == '1': current_cat_idx = 0
                    elif key == '2': current_cat_idx = 1
                    elif key == '3': current_cat_idx = 2
                    elif key == '4': current_cat_idx = 3
                    else: current_cat_idx = -1
                    
                    if current_cat_idx != -1:
                        recorded_audio = [] # Bersihkan buffer lama
                        is_recording = True
                        print("") # Pindah baris
                else:
                    # Berhenti Merekam
                    is_recording = False
                    cat_name = CATEGORIES[current_cat_idx]
                    ts = int(time.time() * 1000)
                    fname = os.path.join(cat_name, f"guru_{cat_name.lower()}_{ts}.wav")
                    
                    if len(recorded_audio) > 0:
                        # Gabungkan semua potongan suara menjadi utuh
                        full_audio = np.concatenate(recorded_audio)
                        
                        # FITUR AUTO-GAIN: Maksimalkan Volume jika suaranya kekecilan!
                        max_val = np.max(np.abs(full_audio))
                        if max_val > 0.001:
                            # Dongkrak volume sampai 95% dari batas maksimal
                            full_audio = full_audio * (0.95 / max_val)
                            
                        # Simpan
                        wav_data = np.clip(full_audio * 32767, -32768, 32767).astype(np.int16)
                        wav.write(fname, SAMPLE_RATE, wav_data)
                        print(f"\n [BERHASIL] ✅ Tersimpan {len(full_audio)/SAMPLE_RATE:.1f} detik (Volume sudah di-Boost) -> {cat_name}!\n")
            
            # Istirahat sebentar agar layar tidak berkedip
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\nProses perekaman selesai. Terima kasih, Guru!")
