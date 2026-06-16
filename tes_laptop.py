import os, sys, time, shutil, json, random
import numpy as np
import tensorflow as tf
import re

try:
    import sounddevice as sd
    import scipy.io.wavfile as wav
    import msvcrt
    import queue
except ImportError:
    print("ERROR: pip install sounddevice scipy")
    sys.exit(1)
    
try:
    import librosa
except ImportError:
    print("ERROR: pip install librosa")
    sys.exit(1)

# Matikan log tensorflow yang mengganggu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

SAMPLE_RATE = 8000
CHUNK_DURATION = 0.1  # Read 100ms at a time for 10Hz updates
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
BUFFER_DURATION = 4.0  # Keep 4 seconds of rolling history
BUFFER_SAMPLES = int(SAMPLE_RATE * BUFFER_DURATION)

# ══════════════════════════════════════════════════════════
# LOAD MODEL & CONFIGURATION
# ══════════════════════════════════════════════════════════
MODEL_PATH = "siren_model_quant.tflite"
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: File model {MODEL_PATH} tidak ditemukan!")
    sys.exit(1)
    
MODEL_H_PATH = r"sirenmaster_main\model.h"
if not os.path.exists(MODEL_H_PATH):
    print(f"ERROR: model.h tidak ditemukan di {MODEL_H_PATH}")
    sys.exit(1)

# Membaca parameter standar dari model.h agar AI sama akuratnya dengan di ESP32
mean, std = [], []
with open(MODEL_H_PATH, 'r') as f:
    content = f.read()
    mean_match = re.search(r'const float MEL_MEAN\[\d+\] PROGMEM = \{([^}]+)\};', content)
    std_match = re.search(r'const float MEL_STD\[\d+\] PROGMEM = \{([^}]+)\};', content)
    if mean_match and std_match:
        mean = np.array([float(x.strip().replace('f', '')) for x in mean_match.group(1).split(',')])
        std = np.array([float(x.strip().replace('f', '')) for x in std_match.group(1).split(',')])

# ══════════════════════════════════════════════════════════
# PERSISTENT STATISTICS FOR EXAMINERS (DASHBOARD DATA)
# ══════════════════════════════════════════════════════════
STATS_FILE = "realtime_stats.json"

def load_stats():
    default_stats = {
        'total': 0,
        'AMBULANCE': {'correct': 0, 'incorrect': 0},
        'FIRETRUCK': {'correct': 0, 'incorrect': 0},
        'NORMAL': {'correct': 0, 'incorrect': 0},
        'POLICE': {'correct': 0, 'incorrect': 0}
    }
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                d = json.load(f)
                d.setdefault('total', 0)
                for cat in CATEGORIES:
                    d.setdefault(cat, {'correct': 0, 'incorrect': 0})
                return d
        except:
            pass
    return default_stats

def save_stats(d):
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(d, f, indent=2)
    except:
        pass

stats = load_stats()
current_prediction = 2  # Default ke NORMAL


print("\n" + "=" * 55)
print("  ALAT TES SIRINE SUPER AI — MIC LAPTOP REAL-TIME")
print("  (Memakai Otak Baru 249x40 Mel-Spectrogram 4 Detik)")
print("=" * 55)
print(f"  Model  : {MODEL_PATH}")
print(f"  Shape  : 4 Detik (249 frame x 40 Mel)")
print("=" * 55)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

# ══════════════════════════════════════════════════════════
# FEATURE EXTRACTION — SAMA PERSIS dengan train_lokal.py
# (3-tap LPF + DC removal + Hamming + FFT manual + Mel FB)
# ══════════════════════════════════════════════════════════
N_FFT      = 256
HOP_LENGTH = 128
N_MELS     = 40

# Pre-compute mel filterbank (sama dengan training)
_hamming = np.hamming(N_FFT).astype(np.float32)
_mel_fb  = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0, fmax=4000)

def extract_features(y, mean, std):
    target_len = BUFFER_SAMPLES  # 32000 samples = 4 detik @ 8kHz
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]

    # 1. 3-tap LPF — sama dengan Arduino & train_lokal.py
    y_lpf = np.convolve(y, [1/3, 1/3, 1/3], mode='same').astype(np.float32)

    # 2. Frame-by-frame: DC removal + Hamming + FFT + Mel
    n_frames = (target_len - N_FFT) // HOP_LENGTH + 1
    log_mel_frames = []
    for i in range(n_frames):
        s = i * HOP_LENGTH
        frame = y_lpf[s:s + N_FFT].copy()
        frame -= frame.mean()                          # DC removal
        frame *= _hamming                              # Hamming window
        fft_out   = np.fft.rfft(frame, n=N_FFT)
        power     = np.abs(fft_out) ** 2              # Power spectrum
        mel_e     = np.dot(_mel_fb, power)            # Mel energies
        log_mel_frames.append(np.log(mel_e + 1e-9))  # Log-Mel

    feat = np.array(log_mel_frames, dtype=np.float32)  # (249, 40)
    feat_scaled = (feat - mean[np.newaxis, :]) / std[np.newaxis, :]
    return feat_scaled

def run_inference(feat_scaled):
    if inp_det['dtype'] == np.int8:
        scale, zero_point = inp_det['quantization']
        feat_quant = np.round(feat_scaled / scale) + zero_point
        feat_input = np.clip(feat_quant, -128, 127).astype(np.int8)
    else:
        feat_input = feat_scaled.astype(np.float32)
        
    feat_input = np.expand_dims(feat_input, axis=0)
    feat_input = np.expand_dims(feat_input, axis=-1)
    
    interpreter.set_tensor(inp_det['index'], feat_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(out_det['index'])
    
    if out_det['dtype'] == np.int8:
        scale, zero_point = out_det['quantization']
        probs = (output_data[0].astype(np.float32) - zero_point) * scale
    else:
        probs = output_data[0]
        
    return probs

# ══════════════════════════════════════════════════════════
# SMART DETECT (Logika Penahan 4.0 Detik)
# ══════════════════════════════════════════════════════════
last_class = 2
consec = 0
confirmed = 2

def smart_detect(raw):
    global last_class, consec, confirmed
    if raw == -1 or raw == 2: # Reset instan jika kembali hening/normal atau di-reset manual
        last_class = 2
        consec = 0
        confirmed = 2
        return 2
        
    if raw == last_class:
        consec += 1
    else:
        last_class = raw
        consec = 1
        
    if consec >= 40: # Konfirmasi setelah 40 frame berturut-turut (~4.0 detik)
        confirmed = last_class
    return confirmed

try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass
os.system('') # Force enable ANSI di Windows CMD
os.system("cls" if os.name == "nt" else "clear") # Bersihkan layar di awal agar rapi

# ANSI 256-Colors (Futuristic Neon Cyberpunk Theme)
C  = '\033[38;5;45m'   # Neon Aqua/Cyan (Borders & Highlights)
G  = '\033[38;5;82m'   # Bright Neon Green (Correct/Normal)
Y  = '\033[38;5;220m'  # Neon Gold/Yellow (Ambulance/Warning)
R  = '\033[38;5;196m'  # Cyber Red (Firetruck/Alert)
B  = '\033[38;5;39m'   # Royal Blue (Police)
M  = '\033[38;5;99m'   # Violet/Purple (Admin/Reset)
W  = '\033[38;5;255m'  # Pure White
GR = '\033[38;5;244m'  # Soft Slate Gray (Secondary info)
RS = '\033[0m'         # Reset
BD = '\033[1m'         # Bold

def get_waveform_bar(rms, length=44):
    if rms < 0.001:
        half = length // 2
        return f"{GR}•{'•' * (half - 1)} {G}🔊 {GR}{'•' * (half - 1)}•{RS}"
        
    level = min(10, int(rms * 1000))
    chars = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    wave = []
    half_len = length // 2
    for i in range(length):
        dist_from_center = abs(i - half_len)
        factor = max(0.0, 1.0 - (dist_from_center / half_len))
        val = int(level * factor * random.uniform(0.7, 1.3))
        val = min(len(chars) - 1, max(0, val))
        
        char = chars[val]
        if char == " ":
            wave.append(f"{GR}•{RS}")
        else:
            if factor > 0.7:
                wave.append(f"{G}{char}{RS}")
            elif factor > 0.4:
                wave.append(f"{Y}{char}{RS}")
            else:
                wave.append(f"{C}{char}{RS}")
    return "".join(wave)

def draw_ui(status_text, conf=None, probs_adj=None, rec_status="", rms=0.0):
    sys.stdout.write("\033[H")
    
    try:
        cols, rows = shutil.get_terminal_size((120, 30))
    except:
        cols, rows = 120, 30
        
    box_w = min(110, max(85, cols - 6))
    box_h = 26
    
    pad_y = max(0, (rows - box_h) // 2)
    pad_x = " " * max(0, (cols - box_w) // 2)
    
    def bprint(line_content):
        # Buang semua karakter ANSI (warna/style) untuk menghitung panjang asli huruf
        clean_text = re.sub(r'\x1b\[[0-9;]*m', '', line_content)
        # Emoji memakan 2 ruang kolom di terminal Windows
        emoji_offset = sum(1 for c in clean_text if c in ['🎧', '🚨', '🔴', '🚑', '🚒', '🚓', '🍃', '👍', '👎', '🔊'])
        vis_len = len(clean_text) + emoji_offset
        
        pad_right = max(0, box_w - 6 - vis_len)
        print(pad_x + f"{C}║{RS}  " + line_content + " " * pad_right + f"  {C}║{RS}")

    print("\n" * pad_y, end="")
    print(pad_x + f"{C}╔" + "═" * (box_w - 2) + f"╗{RS}")
    
    title = "S I R E N   M A S T E R   A I"
    pad_t = (box_w - 2 - len(title)) // 2
    pad_t_r = (box_w - 2 - len(title)) - pad_t
    print(pad_x + f"{C}║{RS}{BD}{C}" + " "*pad_t + title + " "*pad_t_r + f"{RS}{C}║{RS}")
    
    print(pad_x + f"{C}╠" + "═" * (box_w - 2) + f"╣{RS}")
    
    bprint(f"{GR}[ FITUR REKAM GURU OTOMATIS ]{RS}")
    bprint(f"{GR}Jika AI salah tebak, tekan tombol:{RS}")
    bprint(f"{Y}[1] AMBULANCE{RS}   {R}[2] FIRETRUCK{RS}   {G}[3] NORMAL{RS}   {B}[4] POLICE{RS}")
    
    print(pad_x + f"{C}╠" + "═" * (box_w - 2) + f"╣{RS}")
    
    if rec_status != "":
        bprint(f"{R}🔴 RECORDING IN PROGRESS...{RS}")
        bprint(f"{Y}{rec_status}{RS}")
        bprint("")
        bprint("")
        bprint("")
        bprint("")
    else:
        # Tentukan warna status
        color = G if "Aman" in status_text or "Hening" in status_text or "Listening" in status_text or "LISTENING" in status_text else (R if "FIRETRUCK" in status_text else (Y if "AMBULANCE" in status_text else B))
        if status_text == "Menunggu suara...": color = W
        
        status_line = f"{BD}STATUS :{RS} {color}{BD}{status_text}{RS}"
        bprint(status_line)
        bprint("")
        
        if conf is not None and probs_adj is not None:
            # Sisa ruang bar: box_w - 6 (border) - 9 (YAKIN  :) - 8 (100.0% [) - 1 (]) = box_w - 24
            b_w = max(10, box_w - 24)
            bar_len = int(conf * b_w)
            bar = "█" * bar_len + "░" * (b_w - bar_len)
            conf_str = f"{conf*100:5.1f}% [{bar}]"
            bprint(f"{BD}YAKIN  :{RS} {color}{conf_str}{RS}")
            bprint("")
            
            p_ambu = probs_adj[0]*100
            p_fire = probs_adj[1]*100
            p_norm = probs_adj[2]*100
            p_poli = probs_adj[3]*100
            
            bprint(f"{GR}RAW PROBABILITIES:{RS}")
            colored_prob = f"{Y}AMBULANCE: {p_ambu:3.0f}%{RS}   {R}FIRETRUCK: {p_fire:3.0f}%{RS}   {G}NORMAL: {p_norm:3.0f}%{RS}   {B}POLICE: {p_poli:3.0f}%{RS}"
            bprint(colored_prob)
        else:
            wave_str = get_waveform_bar(rms, length=box_w-16)
            bprint(wave_str)
            bprint("")
            bprint("")
            bprint("")
            
    # ══════════════════════════════════════════════════════════
    # DASHBOARD PENGUJIAN REAL-TIME (TABEL MEWAH)
    # ══════════════════════════════════════════════════════════
    print(pad_x + f"{C}╠" + "═" * (box_w - 2) + f"╣{RS}")
    bprint(f"{GR}[ DASHBOARD PENGUJIAN REAL-TIME ]{RS}")
    bprint(f"{C}┌{'─'*16}┬{'─'*12}┬{'─'*12}┬{'─'*9}┐{RS}")
    bprint(f"{C}│{W}{BD} KATEGORI SIRINE{RS}{C}│{W}{BD} BENAR (OK) {RS}{C}│{W}{BD} SALAH (ERR){RS}{C}│{W}{BD} AKURASI {RS}{C}│{RS}")
    bprint(f"{C}├{'─'*16}┼{'─'*12}┼{'─'*12}┼{'─'*9}┤{RS}")
    
    total_correct = 0
    total_incorrect = 0
    col1_map = {
        'AMBULANCE': f" {Y}🚑 AMBULANCE{RS}   ",
        'FIRETRUCK': f" {R}🚒 FIRETRUCK{RS}   ",
        'POLICE':    f" {B}🚓 POLICE{RS}      ",
        'NORMAL':    f" {G}🍃 NORMAL{RS}      "
    }
    for cat in CATEGORIES:
        cor = stats[cat]['correct']
        inc = stats[cat]['incorrect']
        total_correct += cor
        total_incorrect += inc
        tot = cor + inc
        acc_str = f"{(cor / tot * 100):5.1f}%" if tot > 0 else "  -  "
        
        bprint(f"{C}│{RS}{col1_map[cat]}{C}│{RS}{cor:^12d}{C}│{RS}{inc:^12d}{C}│{RS}{acc_str:^9s}{C}│{RS}")
        
    bprint(f"{C}└{'─'*16}┴{'─'*12}┴{'─'*12}┴{'─'*9}┘{RS}")
    
    global_acc = (total_correct / stats['total'] * 100) if stats['total'] > 0 else 0.0
    summary_line = f"  {BD}{W}TOTAL UJI: {stats['total']}{RS}  |  {G}Total Benar: {total_correct}{RS}  |  {R}Total Salah: {total_incorrect}{RS}  |  {Y}Akurasi Global: {global_acc:.1f}%{RS}"
    bprint(summary_line)
    
    if AUTO_MODE:
        bprint(f"  {BD}{C}🤖 MODE OTOMATIS RUNNING...{RS}  |  {M}[R] Reset{RS}  |  [ESC] Keluar")
    else:
        bprint(f"  Tombol Manual: {G}[Y] BENAR (Yes){RS}   {R}[N/T] SALAH (No){RS}   {M}[R] RESET DASHBOARD{RS}")
            
    print(pad_x + f"{C}╚" + "═" * (box_w - 2) + f"╝{RS}")
    footer = "Tekan Ctrl+C untuk keluar."
    print(pad_x + f"{GR}{footer.center(box_w)}{RS}")
    sys.stdout.write("\033[J")
    sys.stdout.flush()

AUTO_MODE = "--auto" in sys.argv

if AUTO_MODE:
    import random
    # Scan all wav files
    all_files = []
    for cat_idx, cat in enumerate(CATEGORIES):
        folder = os.path.join(".", cat)
        if os.path.exists(folder):
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.wav', '.m4a'))]
            # Ambil maksimal 50 sampel acak per kategori untuk pengujian live demo
            random.seed(42)
            if len(files) > 50:
                files = random.sample(files, 50)
            for f in files:
                all_files.append((f, cat_idx))
                
    random.shuffle(all_files)
    
    print("\n" + "=" * 60)
    print("  MEMULAI EVALUASI DATASET BATCH OTOMATIS BERSUARA...")
    print("  (Memutar ke Speaker & Merekam lewat Mic Laptop)")
    print("=" * 60)
    time.sleep(1.5)
    
    try:
        for fpath, true_idx in all_files:
            y, sr = librosa.load(fpath, sr=SAMPLE_RATE)
            target_len = BUFFER_SAMPLES
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)), mode='constant')
            else:
                y = y[:target_len]
                
            # Normalisasi Amplitudo agar volume adil dan terdengar jelas
            max_amp = np.max(np.abs(y))
            y_norm = y / max_amp if max_amp > 1e-4 else y
            
            # Format nama file singkat untuk UI
            fname_short = os.path.basename(fpath)[:25]
            
            # Tampilkan UI "MEMUTAR & MEREKAM" terlebih dahulu
            status_play = f"🔊 PLAY & RECORD: {CATEGORIES[true_idx]} ({fname_short})..."
            draw_ui(status_play, None, None, rms=0.003)
            
            # Putar suara ke speaker dan rekam dari mic secara sinkron selama 4 detik
            recorded_audio = sd.playrec(y_norm, samplerate=SAMPLE_RATE, channels=1)
            sd.wait() # Tunggu sampai pemutaran dan perekaman selesai
            recorded_audio = recorded_audio[:, 0]
            
            # Auto Amplitude Normalization pada hasil rekaman mic
            max_rec_amp = np.max(np.abs(recorded_audio))
            norm_recorded = recorded_audio / max_rec_amp if max_rec_amp > 1e-4 else recorded_audio
            
            # Ekstrak fitur & Jalankan inferensi pada hasil rekaman mic
            feat = extract_features(norm_recorded, mean, std)
            probs = run_inference(feat)
            
            # Posterior Adjustment
            koreksi = np.array([0.80, 1.20, 1.00, 1.00])
            probs_adj = probs * koreksi
            probs_adj = probs_adj / probs_adj.sum()
            
            raw_idx = int(np.argmax(probs_adj))
            conf = float(probs_adj[raw_idx])
            if conf < 0.40:
                raw_idx = 2
                
            final_cls = raw_idx
            current_prediction = final_cls
            
            # Hitung Benar / Salah secara otomatis berdasarkan hasil tebakan mic
            is_correct = (final_cls == true_idx)
            if is_correct:
                stats[CATEGORIES[true_idx]]['correct'] += 1
            else:
                stats[CATEGORIES[true_idx]]['incorrect'] += 1
            stats['total'] += 1
            save_stats(stats)
            
            # Format UI Status
            if is_correct:
                status_str = f"✅ BENAR! {CATEGORIES[true_idx]} ({fname_short})"
            else:
                status_str = f"❌ SALAH! {CATEGORIES[true_idx]} -> {CATEGORIES[final_cls]} ({fname_short})"
                
            # Render UI dengan hasil analisis mic
            rec_rms = float(np.sqrt(np.mean(recorded_audio ** 2)))
            draw_ui(status_str, conf, probs_adj, rms=rec_rms)
            
            # Jeda 2.0 detik agar dosen penguji sempat membaca hasilnya
            # Dibagi menjadi loop kecil agar tetap responsif jika tombol ESC/R ditekan
            break_loop = False
            for _ in range(20):
                time.sleep(0.1)
                if msvcrt.kbhit():
                    break_loop = True
                    break
                    
            if break_loop or msvcrt.kbhit():
                try:
                    key = msvcrt.getch().decode('utf-8').lower()
                except:
                    key = ""
                if key == '\x1b': # ESC
                    break
                elif key == 'r': # RESET
                    stats = {
                        'total': 0,
                        'AMBULANCE': {'correct': 0, 'incorrect': 0},
                        'FIRETRUCK': {'correct': 0, 'incorrect': 0},
                        'NORMAL': {'correct': 0, 'incorrect': 0},
                        'POLICE': {'correct': 0, 'incorrect': 0}
                    }
                    save_stats(stats)
                    draw_ui("Dashboard Direset!", None, None)
                    time.sleep(1.0)
                    
        print("\n✅ Evaluasi Batch Otomatis Selesai!")
        
    except KeyboardInterrupt:
        pass

else:
    # ─── MODE MICROPHONE REAL-TIME ───
    rolling_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)
    samples_filled = 0
    recording_status = ""
    recording_audio = []
    rekam_counter = 0
    
    # State pembekuan untuk verifikasi manual
    frozen = False
    frozen_class = 2
    frozen_conf = 0.0
    frozen_probs_adj = None
    frozen_rms = 0.0

    q = queue.Queue()
    def audio_callback(indata, frames, time, status):
        q.put(indata.copy()[:, 0])

    try:
        # Gambar UI pertama kali
        draw_ui("Menunggu suara...", None, None)
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SAMPLES):
            while True:
                if frozen:
                    # Buang antrean audio baru di background agar tidak lag
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                            
                    # Cek tombol keyboard instan
                    if msvcrt.kbhit():
                        try:
                            key = msvcrt.getch().decode('utf-8').lower()
                        except:
                            key = ""
                            
                        if key == 'y': # BENAR (Yes)
                            stats[CATEGORIES[frozen_class]]['correct'] += 1
                            stats['total'] += 1
                            save_stats(stats)
                            
                            status_conf = f"👍 TERKONFIRMASI BENAR: {CATEGORIES[frozen_class]}!"
                            draw_ui(status_conf, None, None)
                            time.sleep(1.2)
                            
                            # Flush queue
                            while not q.empty():
                                try:
                                    q.get_nowait()
                                except queue.Empty:
                                    break
                            
                            # Bersihkan buffer & penghitung agar tidak memicu deteksi ulang suara lama
                            rolling_buffer.fill(0)
                            samples_filled = 0
                            frozen = False
                            smart_detect(-1)
                            current_prediction = 2
                            
                        elif key in ('t', 'n'): # SALAH (No)
                            stats[CATEGORIES[frozen_class]]['incorrect'] += 1
                            stats['total'] += 1
                            save_stats(stats)
                            
                            status_conf = f"👎 TERKONFIRMASI SALAH: {CATEGORIES[frozen_class]}!"
                            draw_ui(status_conf, None, None)
                            time.sleep(1.2)
                            
                            # Flush queue
                            while not q.empty():
                                try:
                                    q.get_nowait()
                                except queue.Empty:
                                    break
                            
                            # Bersihkan buffer & penghitung agar tidak memicu deteksi ulang suara lama
                            rolling_buffer.fill(0)
                            samples_filled = 0
                            frozen = False
                            smart_detect(-1)
                            current_prediction = 2
                            
                        elif key == 'r': # RESET
                            stats = {
                                'total': 0,
                                'AMBULANCE': {'correct': 0, 'incorrect': 0},
                                'FIRETRUCK': {'correct': 0, 'incorrect': 0},
                                'NORMAL': {'correct': 0, 'incorrect': 0},
                                'POLICE': {'correct': 0, 'incorrect': 0}
                            }
                            save_stats(stats)
                            draw_ui("Dashboard Direset!", None, None)
                            time.sleep(1.0)
                            
                            # Flush queue
                            while not q.empty():
                                try:
                                    q.get_nowait()
                                except queue.Empty:
                                    break
                            
                            # Bersihkan buffer & penghitung agar tidak memicu deteksi ulang suara lama
                            rolling_buffer.fill(0)
                            samples_filled = 0
                            frozen = False
                            smart_detect(-1)
                            current_prediction = 2
                    else:
                        time.sleep(0.05)
                    continue

                # Mengambil 100ms dari queue
                audio = q.get()
                
                # Geser buffer lama, masukkan yang baru (sistem 4 detik berjalan)
                rolling_buffer = np.roll(rolling_buffer, -CHUNK_SAMPLES)
                rolling_buffer[-CHUNK_SAMPLES:] = audio
                
                samples_filled = min(BUFFER_SAMPLES, samples_filled + CHUNK_SAMPLES)
                
                if recording_status != "":
                    recording_audio.append(audio.copy())
                    rekam_counter += 1
                    
                    rec_msg = f"Merekam {recording_status} ({rekam_counter * CHUNK_DURATION:.1f}/10 detik)"
                    draw_ui("", None, None, rec_status=rec_msg)
                    
                    if rekam_counter >= int(10.0 / CHUNK_DURATION):
                        os.makedirs(recording_status, exist_ok=True)
                        filename = f"guru_{recording_status.lower()}_otomatis_{int(time.time())}.wav"
                        filepath = os.path.join(recording_status, filename)
                        
                        full_audio = np.concatenate(recording_audio)
                        max_amp = np.max(np.abs(full_audio))
                        norm_audio = full_audio / max_amp if max_amp > 0 else full_audio
                        wav_data = np.int16(norm_audio * 32767)
                        wav.write(filepath, SAMPLE_RATE, wav_data)
                        
                        recording_status = ""
                        recording_audio = []
                        rekam_counter = 0
                        draw_ui(f"Tersimpan di {filename}!", None, None)
                
                # Fitur Rekam & Dashboard Reset (Tombol Y/N hanya berlaku saat Frozen)
                if msvcrt.kbhit() and recording_status == "":
                    try:
                        key = msvcrt.getch().decode('utf-8').lower()
                    except:
                        key = ""
                    
                    cat_target = ""
                    if key == '1': cat_target = "AMBULANCE"
                    elif key == '2': cat_target = "FIRETRUCK"
                    elif key == '3': cat_target = "NORMAL"
                    elif key == '4': cat_target = "POLICE"
                    
                    # --- RESET DASHBOARD SAJA SAAT LISTENING ---
                    elif key == 'r': # RESET
                        stats = {
                            'total': 0,
                            'AMBULANCE': {'correct': 0, 'incorrect': 0},
                            'FIRETRUCK': {'correct': 0, 'incorrect': 0},
                            'NORMAL': {'correct': 0, 'incorrect': 0},
                            'POLICE': {'correct': 0, 'incorrect': 0}
                        }
                        save_stats(stats)
                    
                    if cat_target != "":
                        recording_status = cat_target
                        recording_audio = []
                        rekam_counter = 0
                        
                        # KOSONGKAN ANTREAN SUARA LAMA YANG MENUMPUK (FLUSH QUEUE)
                        while not q.empty():
                            try:
                                q.get_nowait()
                            except queue.Empty:
                                break
                                
                        draw_ui("", None, None, rec_status=f"Mulai merekam {cat_target}...")
                
                # Cek suara dari 100ms terakhir
                rms = np.sqrt(np.mean(audio ** 2))
                
                # Jika buffer belum terisi penuh 4 detik, kumpulkan data dulu
                if samples_filled < BUFFER_SAMPLES:
                    pct = int(samples_filled / BUFFER_SAMPLES * 100)
                    draw_ui(f"🎧 INITIALIZING BUFFER... ({pct}%)", None, None, rms=rms)
                    continue
                    
                if rms < 0.002: # Diperbesar dari 0.0005 agar tidak sensitif noise statis
                    if recording_status == "":
                        draw_ui(f"🎧 LISTENING... (Hening RMS: {rms:.5f})", None, None, rms=rms)
                    current_prediction = 2 # Default ke NORMAL
                    smart_detect(2)
                    continue
                    
                if recording_status == "":
                    # --- AUTO AMPLITUDE NORMALIZATION ---
                    max_amp = np.max(np.abs(rolling_buffer))
                    normalized_buffer = rolling_buffer / max_amp if max_amp > 1e-4 else rolling_buffer
                    
                    # Ekstrak Fitur 4 Detik Penuh dari buffer yang dinormalisasi!
                    feat = extract_features(normalized_buffer, mean, std)
                    probs = run_inference(feat)
                    
                    # --- POSTERIOR ADJUSTMENT (Koreksi Bias AI) v3 [AMAN] ---
                    koreksi = np.array([0.80, 1.20, 1.00, 1.00])
                    probs_adj = probs * koreksi
                    probs_adj = probs_adj / probs_adj.sum()
                    
                    raw_idx = int(np.argmax(probs_adj))
                    conf = float(probs_adj[raw_idx])
                    
                    # Syarat minimal yakin 40%
                    if conf < 0.40:
                        raw_idx = 2 # Anggap NORMAL
                        
                    final_cls = smart_detect(raw_idx)
                    current_prediction = final_cls
                    
                    if final_cls == 2: # NORMAL
                        status_str = f"🎧 LISTENING... (Aman - RMS: {rms:.4f})"
                        draw_ui(status_str, conf, probs_adj, rms=rms)
                    else:
                        # PEMICU PEMBEKUAN (SIREN TERDETEKSI!)
                        frozen = True
                        frozen_class = final_cls
                        frozen_conf = conf
                        frozen_probs_adj = probs_adj
                        frozen_rms = rms
                        
                        status_str = f"🚨 {CATEGORIES[final_cls]} DETECTED! (RMS: {rms:.4f}) - MENUNGGU VERIFIKASI..."
                        draw_ui(status_str, conf, probs_adj, rms=rms)

    except KeyboardInterrupt:
        os.system("cls" if os.name == "nt" else "clear")
        print("Selesai! Sampai jumpa.")
