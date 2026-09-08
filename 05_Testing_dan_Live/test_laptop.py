"""
test_laptop.py — Siren Detector untuk Laptop (CMD / Terminal)
==============================================================
Fitur:
  - Sliding window 4 detik, update setiap 0.5 detik
  - EMA accumulator untuk deteksi stabil
  - DSP pipeline IDENTIK dengan train_lokal.py & ESP32
  - Hot-swap online learning (tekan 1-4 untuk koreksi)
"""
import os
import sys
import numpy as np
import librosa
import tensorflow as tf
import queue
import threading
import time

# Paksa stdout menggunakan encoding UTF-8 agar karakter emoji / block bar berjalan lancar
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

try:
    import sounddevice as sd
    import scipy.io.wavfile as wav
    import msvcrt
except ImportError:
    print("[ERROR] Modul 'sounddevice' atau 'scipy' belum terinstal.")
    print("Silakan buka terminal/Command Prompt dan ketik: pip install sounddevice scipy")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════
#  KONFIGURASI (HARUS SAMA PERSIS DENGAN train_lokal.py & ESP32)
# ════════════════════════════════════════════════════════════════
CACHE_PATH = "siren_40x249_melspec_cache_lokal.npz"
MODEL_PATH = "siren_model_quant.tflite"

SAMPLE_RATE = 8000
DURATION = 4.0
N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40
N_FMAX = 4000

UPDATE_INTERVAL = 0.5  # Update UI setiap 0.5 detik
CHUNK_SAMPLES = int(SAMPLE_RATE * UPDATE_INTERVAL)
BUFFER_SAMPLES = int(SAMPLE_RATE * DURATION)
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

# ════════════════════════════════════════════════════════════════
#  CEK FILE YANG DIBUTUHKAN
# ════════════════════════════════════════════════════════════════
if not os.path.exists(CACHE_PATH) or not os.path.exists(MODEL_PATH) or not os.path.exists("siren_scaler.npz"):
    print("[ERROR] File model, cache, atau scaler tidak ditemukan!")
    print(f"  Pastikan file ini ada: {CACHE_PATH}, {MODEL_PATH}, dan siren_scaler.npz")
    print("  Solusi: Harap jalankan 'python train_lokal.py' terlebih dahulu untuk melakukan training dan generate file tersebut.")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════
#  DSP PIPELINE — 100% IDENTIK DENGAN train_lokal.py
# ════════════════════════════════════════════════════════════════
# Pre-compute filterbank & window sekali saja (efisien)
_hamming_window = np.hamming(N_FFT).astype(np.float32)
_mel_filterbank = librosa.filters.mel(
    sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=N_FMAX
)

def extract_melspec(y):
    """Ekstraksi Log-Mel Spectrogram IDENTIK dengan train_lokal.py:
    1. Pad/trim ke 4 detik (32000 sampel)
    2. 3-tap moving average LPF
    3. Per-frame: DC removal → Hamming → FFT → Power → Mel → Log
    """
    target_len = BUFFER_SAMPLES
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]

    # 1. 3-tap moving average filter (Low Pass Filter)
    y_smoothed = np.convolve(y, [1/3, 1/3, 1/3], mode='same')

    # 2. Frame-by-frame processing
    n_frames = (target_len - N_FFT) // HOP_LENGTH + 1
    log_mel_frames = np.empty((n_frames, N_MELS), dtype=np.float32)

    for frame_idx in range(n_frames):
        start = frame_idx * HOP_LENGTH
        frame_data = y_smoothed[start:start + N_FFT].copy()

        # DC removal
        frame_data -= np.mean(frame_data)

        # Hamming windowing
        frame_windowed = frame_data * _hamming_window

        # Power spectrum
        fft_complex = np.fft.rfft(frame_windowed, n=N_FFT)
        power_spec = np.abs(fft_complex) ** 2

        # Mel energies
        mel_energies = np.dot(_mel_filterbank, power_spec)

        # Log energy (with 1e-9 floor to match logf(energy + 1e-9f))
        log_mel_frames[frame_idx] = np.log(mel_energies + 1e-9)

    return log_mel_frames  # shape: (249, 40)

# ════════════════════════════════════════════════════════════════
#  LOAD MODEL & NORMALISASI
# ════════════════════════════════════════════════════════════════
model_lock = threading.Lock()
global_mean, global_std = None, None
interpreter, input_details, output_details = None, None, None

def load_brain():
    """Reload model DAN normalisasi dari cache terbaru."""
    global global_mean, global_std, interpreter, input_details, output_details
    with model_lock:
        with np.load("siren_scaler.npz", allow_pickle=True) as data:
            global_mean = data['global_mean']
            global_std = data['global_std']

        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

print("[*] Memuat Memori AI...")
load_brain()

# ════════════════════════════════════════════════════════════════
#  STATE GLOBAL
# ════════════════════════════════════════════════════════════════
audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)
q = queue.Queue()

recent_chunks = []
PAUSED = False
RECORDING_IN_PROGRESS = False

# EMA Accumulator: [AMBULANCE, FIRETRUCK, NORMAL, POLICE]
ema_probs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
BUFFER_WARMUP = int(DURATION / UPDATE_INTERVAL)  # 8 update = 4 detik
loud_chunks_count = 0

# Display state
current_rms = 0.0
current_label = "NORMAL"
current_prob = 1.0
locked_class = None


# ════════════════════════════════════════════════════════════════
#  AUDIO CALLBACK
# ════════════════════════════════════════════════════════════════
def audio_callback(indata, frames, time_info, status):
    q.put(indata.copy()[:, 0])

# ════════════════════════════════════════════════════════════════
#  DISPLAY THREAD (update setiap 100ms)
# ════════════════════════════════════════════════════════════════
def display_thread():
    global current_rms, current_label, current_prob
    spinner = ['|', '/', '-', '\\']
    idx = 0
    
    SHORT_LABELS = {
        'NORMAL': 'SAFE',
        'AMBULANCE': 'AMBU',
        'FIRETRUCK': 'FIRE',
        'POLICE': 'POL'
    }
    
    while True:
        spin = spinner[idx % 4]
        # Perkecil bar volume ke 6 karakter agar hemat kolom (12 kolom visual)
        bar_len = min(int(current_rms / 300.0 * 6), 6)
        vol_bar = '█' * bar_len + '░' * (6 - bar_len)
        
        if RECORDING_IN_PROGRESS:
            time.sleep(0.1)
            continue
            
        lbl = current_label
        if "MENILAI" in lbl:
            # Contoh: "MENILAI FIRETRUCK..." -> "?FIRE"
            parsed = False
            for k, v in SHORT_LABELS.items():
                if k in lbl:
                    lbl = f"?{v}"
                    parsed = True
                    break
            if not parsed:
                lbl = "??"
        else:
            lbl = SHORT_LABELS.get(lbl, lbl)
            
        if PAUSED:
            line = f"[{spin}] PAUSED | Vol: {vol_bar} ({current_rms:.0f})"
        else:
            line = f"[{spin}] {lbl} ({current_prob * 100:.0f}%) | Vol: {vol_bar} ({current_rms:.0f})"
            
        try:
            # Hapus sisa baris secara bersih dan cetak satu baris pendek
            sys.stdout.write(f"\r\033[K{line}")
            sys.stdout.flush()
        except:
            # Fallback jika tidak mendukung ANSI
            ascii_bar = '#' * bar_len + '-' * (6 - bar_len)
            if PAUSED:
                line_ascii = f"[{spin}] PAUSED | Vol: {ascii_bar} ({current_rms:.0f})"
            else:
                line_ascii = f"[{spin}] {lbl} ({current_prob * 100:.0f}%) | Vol: {ascii_bar} ({current_rms:.0f})"
            try:
                sys.stdout.write(f"\r{line_ascii:<45}")
                sys.stdout.flush()
            except:
                pass
        idx += 1
        time.sleep(0.1)


# ════════════════════════════════════════════════════════════════
#  AUDIO PROCESSING THREAD (inti deteksi)
# ════════════════════════════════════════════════════════════════
def process_audio():
    global recent_chunks, current_rms, current_label, current_prob, locked_class
    global audio_buffer, ema_probs, loud_chunks_count

    while True:
        if PAUSED:
            time.sleep(0.1)
            continue

        # Pastikan tidak ada lag (buang data audio usang jika menumpuk di antrean)
        while q.qsize() > 1:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        
        audio_data = q.get()

        # ── SLIDING WINDOW BUFFER ──
        audio_buffer = np.roll(audio_buffer, -len(audio_data))
        audio_buffer[-len(audio_data):] = audio_data

        # Simpan chunk terakhir untuk fitur admin (simpan 8 detik)
        recent_chunks.append(audio_data.copy())
        if len(recent_chunks) > int(8.0 / UPDATE_INTERVAL):
            recent_chunks.pop(0)

        # RMS dari chunk terbaru (untuk volume meter)
        rms = np.sqrt(np.mean(audio_data**2))
        current_rms = rms * 32768

        # Suara terlalu pelan → Langsung kembali ke NORMAL secara instan
        # Threshold diturunkan ke 0.01 agar suara HP dekat mic laptop bisa terdeteksi
        if rms < 0.01:
            loud_chunks_count = 0
            locked_class = None
            ema_probs[:] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            current_label = "NORMAL"
            current_prob = 1.0
            continue

        # ── TILING ACTIVE SOUND SEGMENT ──
        # Menghindari distorsi dari silence di buffer saat awal suara terdengar.
        # Kita tile (ulangi) suara aktif hingga memenuhi 4 detik.
        loud_chunks_count += 1
        active_chunks = min(loud_chunks_count, BUFFER_WARMUP)
        active_samples = active_chunks * CHUNK_SAMPLES
        active_audio = audio_buffer[-active_samples:]

        repeats = int(np.ceil(BUFFER_SAMPLES / active_samples))
        buffer_copy = np.tile(active_audio, repeats)[:BUFFER_SAMPLES]

        # ── EKSTRAKSI FITUR (Sinyal mentah sesuai aslinya!) ──
        feat = extract_melspec(buffer_copy)

        # ── NORMALISASI Z-SCORE ──
        with model_lock:
            cur_mean = global_mean.copy()
            cur_std = global_std.copy()

        feat_scaled = (feat - cur_mean) / cur_std
        feat_scaled = feat_scaled.astype(np.float32)
        feat_scaled = np.expand_dims(feat_scaled, axis=0)     # → (1, 249, 40)
        feat_scaled = np.expand_dims(feat_scaled, axis=-1)    # → (1, 249, 40, 1)

        # ── INFERENSI TFLITE ──
        with model_lock:
            input_scale, input_zero_point = input_details[0]['quantization']
            if input_details[0]['dtype'] == np.int8:
                feat_quant = np.round(feat_scaled / input_scale) + input_zero_point
                feat_quant = np.clip(feat_quant, -128, 127).astype(np.int8)
            else:
                feat_quant = feat_scaled

            interpreter.set_tensor(input_details[0]['index'], feat_quant)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_scale, output_zero_point = output_details[0]['quantization']
            if output_details[0]['dtype'] == np.int8:
                probs = (output_data[0].astype(np.float32) - output_zero_point) * output_scale
            else:
                probs = output_data[0]

        # ── NORMALISASI OUTPUT ──
        # Model sudah punya softmax di layer terakhir, jadi output
        # setelah dequantisasi sudah berupa probabilitas (0-1).
        # Kita hanya perlu clamp & normalize untuk menangani error kuantisasi kecil.
        probs_clean = np.clip(probs, 0.0, None)  # Buang nilai negatif dari noise kuantisasi
        total = np.sum(probs_clean)
        if total > 0:
            probs_norm = probs_clean / total
        else:
            probs_norm = np.array([0.0, 0.0, 1.0, 0.0])  # Default NORMAL

        # ── EMA ACCUMULATOR (deteksi stabil) ──
        # Alpha dikecilkan dari 0.30 menjadi 0.15 agar respon lebih lambat dan stabil, tidak mudah tertipu noise sesaat
        EMA_ALPHA = 0.15
        ema_probs[:] = (1 - EMA_ALPHA) * ema_probs + EMA_ALPHA * probs_norm

        # Re-normalisasi
        ema_sum = np.sum(ema_probs)
        if ema_sum > 0:
            ema_probs[:] = ema_probs / ema_sum

        # Kelas terbaik dari akumulator
        best_idx = int(np.argmax(ema_probs))
        best_prob = float(ema_probs[best_idx])
        best_label = CATEGORIES[best_idx]

        # ── THRESHOLD & LOCK-ON ──
        # Diperketat menjadi 80% agar tidak asal tebak kelas
        SIREN_THRESHOLD = 0.80
        OVERRIDE_THRESHOLD = 0.95

        if locked_class is not None:
            locked_idx = CATEGORIES.index(locked_class)
            if ema_probs[locked_idx] < SIREN_THRESHOLD:
                locked_class = None

        if locked_class is None:
            # Belum terkunci: cari sirine baru yang meyakinkan
            if best_label != 'NORMAL' and best_prob >= SIREN_THRESHOLD:
                locked_class = best_label
                current_label = locked_class
                current_prob = best_prob
            elif best_label != 'NORMAL' and best_prob > 0.40:
                # UX Baru: Tampilkan proses "Pikir Dulu" AI
                current_label = f"MENILAI {best_label}..."
                current_prob = best_prob
            else:
                current_label = 'NORMAL'
                current_prob = float(ema_probs[2])
        else:
            # Sudah terkunci: tetap gunakan locked_class kecuali ada override dari kelas sirine lain yang sangat yakin (>= 90%)
            if best_label != 'NORMAL' and best_label != locked_class and best_prob >= OVERRIDE_THRESHOLD:
                locked_class = best_label
                # Atur ulang EMA agar langsung berpihak ke kelas override
                ema_probs[:] = 0.0
                ema_probs[best_idx] = 1.0
                current_label = locked_class
                current_prob = best_prob
            else:
                # Pertahankan kelas terkunci
                locked_idx = CATEGORIES.index(locked_class)
                current_label = locked_class
                current_prob = float(ema_probs[locked_idx])


# ════════════════════════════════════════════════════════════════
#  MAIN: Jalankan semua thread
# ════════════════════════════════════════════════════════════════
threading.Thread(target=display_thread, daemon=True).start()
threading.Thread(target=process_audio, daemon=True).start()

print(f"\n[*] Mengaktifkan Mikrofon Laptop (Sample Rate: {SAMPLE_RATE} Hz)...")
print("Silakan putar suara sirine dari HP Anda dan dekatkan ke mikrofon laptop.")
print("=" * 60)
print("  [ FITUR ADMIN KUMPUL DATASET ]")
print("  Jika AI salah tebak, Anda bisa tekan angka di keyboard untuk")
print("  menyimpan 8 detik suara terakhir secara otomatis ke dalam folder:")
print("    1 -> Simpan sebagai AMBULANCE")
print("    2 -> Simpan sebagai FIRETRUCK")
print("    3 -> Simpan sebagai NORMAL/JALANAN")
print("    4 -> Simpan sebagai POLICE")
print("  Tekan Ctrl+C untuk berhenti program.")
print("=" * 60 + "\n")

# Buat folder untuk simpan koreksi
SAVE_DIR = "TAMBAH_DATASET"
for cat in CATEGORIES:
    os.makedirs(os.path.join(SAVE_DIR, cat), exist_ok=True)

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        callback=audio_callback, blocksize=CHUNK_SAMPLES):
        while True:
            if msvcrt.kbhit():
                try:
                    key = msvcrt.getch().decode('utf-8').lower()
                except:
                    key = ""
                cat_name = None
                if key == '1': cat_name = 'AMBULANCE'
                elif key == '2': cat_name = 'FIRETRUCK'
                elif key == '3': cat_name = 'NORMAL'
                elif key == '4': cat_name = 'POLICE'

                if cat_name:
                    PAUSED = True
                    time.sleep(0.1)
                    
                    # Kosongkan antrean audio yang lama agar kita benar-benar merekam dari 0 detik ke depan
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                            
                    sys.stdout.write(f"\r\n[+] MEREKAM 10 DETIK UNTUK: {cat_name}... HARAP BUNYIKAN SUARA SEKARANG!\n")
                    RECORDING_IN_PROGRESS = True
                    
                    recorded_chunks = []
                    target_chunks = int(10.0 / UPDATE_INTERVAL) # 10 detik = 20 chunks
                    for i in range(target_chunks):
                        chunk = q.get() # Menunggu chunk mikrofon yang masuk secara live
                        recorded_chunks.append(chunk)
                        
                        # Hitung volume live agar tidak terlihat freeze (nyangkut)
                        rms = np.sqrt(np.mean(chunk**2))
                        live_vol = rms * 32768
                        
                        # Tampilkan progress bar dengan volume meter live
                        progress = int((i + 1) / target_chunks * 20)
                        bar = '█' * progress + '░' * (20 - progress)
                        sys.stdout.write(f"\r    Merekam: [{bar}] {(i+1)*UPDATE_INTERVAL:.1f} dtk | Vol: {live_vol:5.0f}    ")
                        sys.stdout.flush()
                        
                    sys.stdout.write("\n[+] Perekaman selesai. Menyimpan dan melatih AI...\n")
                    
                    combined_audio = np.concatenate(recorded_chunks)
                    
                    # OPTIMASI KUALITAS SUARA (Agar tidak mendem)
                    # 1. Hapus DC Offset (dengung statis hardware mikrofon)
                    combined_audio = combined_audio - np.mean(combined_audio)
                    
                    # 2. Peak Normalization ke 95% agar suaranya lantang dan jernih tapi tidak pecah
                    max_amp = np.max(np.abs(combined_audio))
                    if max_amp > 0:
                        combined_audio = combined_audio * (0.95 / max_amp)
                        
                    wav_data = np.int16(np.clip(combined_audio, -1.0, 1.0) * 32767)

                    filename = f"tambah_{int(time.time())}.wav"
                    filepath = os.path.join(SAVE_DIR, cat_name, filename)
                    wav.write(filepath, SAMPLE_RATE, wav_data)

                    # Online Learning
                    try:
                        import subprocess
                        subprocess.run([sys.executable, "fine_tune.py", filepath, cat_name], check=True)
                        load_brain()
                        current_label = "NORMAL"
                        current_prob = 1.0
                        loud_chunks_count = 0
                        # Reset EMA ke NORMAL setelah hot-swap
                        ema_probs[:] = np.array([0.0, 0.0, 1.0, 0.0])
                        sys.stdout.write(f"[✅] HOT-SWAP BERHASIL! Melanjutkan deteksi...{' '*20}\n")

                        # Kosongkan antrean audio yang menumpuk
                        while not q.empty():
                            try:
                                q.get_nowait()
                            except queue.Empty:
                                break

                    except Exception as e:
                        sys.stdout.write(f"\r[!] Gagal melatih: {e}\n")

                    RECORDING_IN_PROGRESS = False
                    PAUSED = False
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\n[*] Program dihentikan.")
except Exception as e:
    print(f"\n[!] Terjadi kesalahan: {e}")
