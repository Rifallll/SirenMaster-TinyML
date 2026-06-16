import os
import sys
# Paksa stdout menggunakan encoding UTF-8 agar karakter emoji / block bar berjalan lancar
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
import numpy as np
import librosa
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']
CACHE_PATH = "siren_40x249_melspec_cache_lokal.npz"   # ← cache baru (249 frame)
TFLITE_PATH = "siren_model_quant.tflite"
OUTPUT_HEADER = r"C:\Users\ASUS\Videos\DATASET\sirenmaster_main\model.h"

SAMPLE_RATE = 8000
DURATION = 4.0                                         # ← 4 detik sesuai model baru
CHUNK_SAMPLES = int(SAMPLE_RATE * DURATION)

N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40

def extract_melspec(y):
    """Ekstraksi fitur persis seperti di train_lokal.py (249x40)."""
    target_len = CHUNK_SAMPLES
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]

    # 3-tap moving average LPF (sama persis dengan C++ DSP)
    y_smoothed = np.convolve(y, [1/3, 1/3, 1/3], mode='same')

    n_frames = (target_len - N_FFT) // HOP_LENGTH + 1
    hamming = np.hamming(N_FFT)
    mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000)

    log_mel_frames = []
    for frame in range(n_frames):
        start = frame * HOP_LENGTH
        frame_data = y_smoothed[start:start+N_FFT].copy()
        frame_data -= np.mean(frame_data)      # DC removal
        frame_windowed = frame_data * hamming  # Hamming window
        fft_complex = np.fft.rfft(frame_windowed, n=N_FFT)
        power_spec = np.abs(fft_complex) ** 2
        mel_energies = np.dot(mel_fb, power_spec)
        log_mel = np.log(mel_energies + 1e-9)
        log_mel_frames.append(log_mel)

    return np.array(log_mel_frames, dtype=np.float32)  # shape: (249, 40)


def augment_and_extract(y, sr, cat_idx):
    """Buat banyak variasi dari satu rekaman."""
    samples_X = []
    samples_y = []

    # Sliding window dengan overlap 50%
    hop = CHUNK_SAMPLES // 2
    for start in range(0, max(1, len(y) - CHUNK_SAMPLES + 1), hop):
        chunk = y[start:start+CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)), mode='constant')

        rms = np.sqrt(np.mean(chunk**2))
        if rms < 0.001:
            continue

        # Normalisasi volume
        max_val = np.max(np.abs(chunk))
        if max_val > 1e-6:
            chunk = chunk / max_val

        # Asli
        samples_X.append(extract_melspec(chunk))
        samples_y.append(cat_idx)

        # Gaussian noise ringan
        noisy = chunk + np.random.normal(0, 0.003, len(chunk))
        samples_X.append(extract_melspec(noisy))
        samples_y.append(cat_idx)

        # Low-pass filter (simulasi suara jauh)
        muffled = np.convolve(chunk, np.ones(5)/5, mode='same')
        samples_X.append(extract_melspec(muffled))
        samples_y.append(cat_idx)

    return samples_X, samples_y


def main():
    if len(sys.argv) < 3:
        print("[ERROR] Usage: python fine_tune.py <wav_path> <category_name>")
        sys.exit(1)

    wav_path = sys.argv[1]
    cat_name = sys.argv[2]

    if cat_name not in CATEGORIES:
        print(f"[ERROR] Kategori tidak dikenal: {cat_name}. Pilihan: {CATEGORIES}")
        sys.exit(1)

    cat_idx = CATEGORIES.index(cat_name)
    print(f"\n[HOT-SWAP] Fine-Tuning untuk: {cat_name}...", flush=True)

    # 1. Load & augment audio baru
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    X_new, y_new = augment_and_extract(y, sr, cat_idx)

    if len(X_new) == 0:
        print("[!] Audio terlalu pendek atau senyap. Tidak ada sampel yang diekstrak.")
        sys.exit(0)

    X_new = np.array(X_new, dtype=np.float32)
    y_new = np.array(y_new, dtype=np.int32)
    print(f"  [+] {len(X_new)} sampel diekstrak dari rekaman baru.", flush=True)

    # 2. Load lightweight scaler and replay buffer
    SCALER_PATH = "siren_scaler.npz"
    REPLAY_PATH = "siren_replay_buffer.npz"

    if not os.path.exists(SCALER_PATH) or not os.path.exists(REPLAY_PATH):
        print(f"[ERROR] Scaler atau Replay Buffer tidak ditemukan.")
        print("  Jalankan dulu: .venv\\Scripts\\python.exe train_lokal.py")
        sys.exit(1)

    print("  [+] Membaca replay buffer...", flush=True)
    with np.load(REPLAY_PATH, allow_pickle=True) as data:
        X_replay = data['X_replay']
        y_replay = data['y_replay']

    with np.load(SCALER_PATH, allow_pickle=True) as data:
        global_mean = data['global_mean']
        global_std = data['global_std']

    # Simpan data baru ke replay buffer untuk online learning berikutnya
    X_replay_new = np.concatenate([X_replay, X_new], axis=0)
    y_replay_new = np.concatenate([y_replay, y_new], axis=0)
    np.savez_compressed(REPLAY_PATH, X_replay=X_replay_new, y_replay=y_replay_new)
    print(f"  [+] Replay buffer diperbarui: {len(X_replay_new)} total sampel.", flush=True)

    # Gabung replay + data baru (duplikat data baru 5x agar fokus)
    X_train = np.concatenate([X_replay] + [X_new]*5, axis=0)
    y_train = np.concatenate([y_replay] + [y_new]*5, axis=0)

    # 3. Normalisasi menggunakan scaler global
    X_scaled = (X_train - global_mean[np.newaxis, np.newaxis, :]) / global_std[np.newaxis, np.newaxis, :]
    X_scaled = X_scaled[..., np.newaxis]  # → (batch, 249, 40, 1)

    # Shuffle
    perm = np.random.permutation(len(X_scaled))
    X_scaled = X_scaled[perm]
    y_train   = y_train[perm]

    # 4. Fine-Tune model TFLite → perlu load model Keras (.h5) terlebih dulu
    # Cek apakah ada file Keras h5 (model sebelum quantisasi)
    KERAS_MODEL_PATH = "siren_classifier_model.h5"
    if not os.path.exists(KERAS_MODEL_PATH):
        print("[!] File model Keras (.h5) tidak ditemukan. Fine-tuning dilewati.")
        print("    Simpan dulu audio ini ke folder NORMAL, lalu jalankan train_lokal.py penuh.")
        # Pindahkan file WAV ke folder NORMAL dataset agar ikut training berikutnya
        dst = os.path.join(r"C:\Users\ASUS\Videos\DATASET\NORMAL", os.path.basename(wav_path))
        if not os.path.exists(dst):
            import shutil
            shutil.copy2(wav_path, dst)
            print(f"    [+] Audio disimpan ke: {dst}")
        sys.exit(0)

    print("  [+] Melatih ulang AI (Micro Fine-Tuning)...", flush=True)
    model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),  # LR sangat kecil
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(X_scaled, y_train, epochs=5, batch_size=16, verbose=0)
    acc = history.history['accuracy'][-1]
    print(f"  [+] Training selesai. Akurasi fine-tune: {acc*100:.1f}%", flush=True)
    model.save(KERAS_MODEL_PATH)

    # 5. Re-Quantize & Export TFLite
    print("  [+] Mengekspor model TFLite baru...", flush=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    # Kalibrasikan dengan data baru
    X_calib = X_scaled[:min(200, len(X_scaled))].astype(np.float32)
    def rep_dataset():
        for sample in X_calib:
            yield [sample[np.newaxis, ...]]
    converter.representative_dataset = rep_dataset

    tflite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"  [+] Model TFLite diperbarui ({len(tflite_model)/1024:.1f} KB).", flush=True)

    # 6. Export to C++ Header model.h to propagate hot-swap to microcontroller
    print(f"  [+] Mengekspor model TFLite baru ke header C++: {OUTPUT_HEADER}...", flush=True)
    try:
        # Re-compute windowing and filterbanks to ensure complete correctness
        hamming_win = np.hamming(N_FFT).astype(np.float32)
        mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000).T
        n_frames = (int(SAMPLE_RATE * DURATION) - N_FFT) // HOP_LENGTH + 1

        with open(OUTPUT_HEADER, 'w') as f_h:
            f_h.write("/*\n")
            f_h.write(" * model.h - Siren Classifier Deployment Header (Hot-swapped)\n")
            f_h.write(" * Generated automatically during online hot-swap fine-tuning\n")
            f_h.write(" */\n\n")
            f_h.write("#ifndef MODEL_H\n")
            f_h.write("#define MODEL_H\n\n")
            f_h.write("#include <pgmspace.h>\n\n")
            
            f_h.write(f"#define SAMPLE_RATE_HZ   {SAMPLE_RATE}\n")
            f_h.write(f"#define N_FFT_SIZE       {N_FFT}\n")
            f_h.write(f"#define N_HOP_LENGTH     {HOP_LENGTH}\n")
            f_h.write(f"#define N_MEL_FILTERS    {N_MELS}\n")
            f_h.write(f"#define N_TIME_FRAMES    {n_frames}\n")
            f_h.write(f"#define N_FFT_BINS       {N_FFT // 2 + 1}\n")
            f_h.write(f"#define AUDIO_SAMPLES    {int(SAMPLE_RATE * DURATION)}\n")
            f_h.write(f"#define NUM_CLASSES      {len(CATEGORIES)}\n\n")
            
            # Hamming window
            f_h.write(f"const float HAMMING_WINDOW[{N_FFT}] PROGMEM = {{\n")
            for i, val in enumerate(hamming_win):
                if i % 8 == 0:
                    f_h.write("  ")
                f_h.write(f"{val:.8f}f")
                if i < N_FFT - 1:
                    f_h.write(", ")
                if (i + 1) % 8 == 0 or i == N_FFT - 1:
                    f_h.write("\n")
            f_h.write("};\n\n")
            
            # Mel filterbank
            f_h.write(f"const float MEL_FILTERBANK[{N_MELS}][{N_FFT//2+1}] PROGMEM = {{\n")
            for m in range(N_MELS):
                f_h.write("  {")
                row = mel_fb[:, m]
                f_h.write(", ".join([f"{val:.8f}f" for val in row]))
                f_h.write("}")
                if m < N_MELS - 1:
                    f_h.write(",\n")
                else:
                    f_h.write("\n")
            f_h.write("};\n\n")
            
            # Scaling parameters
            f_h.write(f"const float MEL_MEAN[{N_MELS}] PROGMEM = {{\n  ")
            f_h.write(", ".join([f"{val:.8f}f" for val in global_mean]))
            f_h.write("\n};\n\n")
            
            f_h.write(f"const float MEL_STD[{N_MELS}] PROGMEM = {{\n  ")
            f_h.write(", ".join([f"{val:.8f}f" for val in global_std]))
            f_h.write("\n};\n\n")
            
            # Quantized model data
            f_h.write("// Align model array for TFLite Micro\n")
            f_h.write("#ifdef __has_attribute\n")
            f_h.write("#define MODEL_ALIGN __attribute__((aligned(4)))\n")
            f_h.write("#else\n")
            f_h.write("#define MODEL_ALIGN\n")
            f_h.write("#endif\n\n")
            
            f_h.write(f"const unsigned int siren_model_data_len = {len(tflite_model)};\n\n")
            f_h.write("const unsigned char siren_model_data[] MODEL_ALIGN = {\n")
            for i, val in enumerate(tflite_model):
                if i % 12 == 0:
                    f_h.write("  ")
                f_h.write(f"0x{val:02x}")
                if i < len(tflite_model) - 1:
                    f_h.write(", ")
                if (i + 1) % 12 == 0 or i == len(tflite_model) - 1:
                    f_h.write("\n")
            f_h.write("};\n\n")
            f_h.write("#endif // MODEL_H\n")
        print(f"  [SUCCESS] C++ Header {OUTPUT_HEADER} updated successfully.", flush=True)
    except Exception as e:
        print(f"  [ERROR] Failed to update C++ Header: {e}", flush=True)

    print("[SUCCESS] HOT-SWAP SELESAI! AI sudah diperbarui.", flush=True)


if __name__ == "__main__":
    main()
