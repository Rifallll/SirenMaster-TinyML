import os
import librosa
import numpy as np

dataset_dir = r"C:\Users\ASUS\Videos\DATASET"
classes = ["AMBULANCE", "FIRETRUCK", "NORMAL", "POLICE"]

print("=== DEEP AUDIT KUALITAS SUARA DATASET ===")
print("Menganalisis tingkat kebisingan, kesunyian, dan distorsi...\n")

for c in classes:
    folder = os.path.join(dataset_dir, c)
    if not os.path.exists(folder):
        continue
    
    files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    if len(files) == 0:
        continue
    
    # Ambil sampel 100 file acak per kelas untuk kecepatan
    np.random.seed(42)
    sample_files = list(np.random.choice(files, min(100, len(files)), replace=False))
    
    silent_count = 0
    clipping_count = 0
    rms_values = []
    
    for f in sample_files:
        try:
            path = os.path.join(folder, f)
            y, sr = librosa.load(path, sr=16000, duration=3.0)  # Load max 3 detik pertama
            
            if len(y) == 0:
                continue
                
            max_amp = np.max(np.abs(y))
            rms = np.sqrt(np.mean(y**2))
            rms_values.append(rms)
            
            # Deteksi file yang nyaris tanpa suara (terlalu sunyi)
            if max_amp < 0.05:
                silent_count += 1
                
            # Deteksi file yang over-amplified (pecah/distorsi)
            if max_amp > 0.99:
                clipping_count += 1
                
        except Exception:
            pass
            
    if rms_values:
        avg_rms = np.mean(rms_values)
        print(f"[{c}] -> Dianalisis: {len(rms_values)} file")
        print(f"  - Rata-rata Volume (RMS) : {avg_rms:.4f}")
        print(f"  - File Terlalu Sunyi     : {silent_count} file ({silent_count/len(rms_values)*100:.1f}%)")
        print(f"  - File Distorsi/Pecah    : {clipping_count} file ({clipping_count/len(rms_values)*100:.1f}%)")
        
        if c == "NORMAL":
            if silent_count > 20:
                print("  [!] WARNING: Terlalu banyak file sunyi di kelas NORMAL! AI bisa salah tebak jika mendengar keramaian jalan.")
            if avg_rms < 0.05:
                print("  [!] WARNING: Volume rata-rata kelas NORMAL terlalu kecil. Perbanyak suara klakson/jalan raya!")
                
        if c != "NORMAL":
            if clipping_count > 50:
                print(f"  [!] WARNING: Terlalu banyak distorsi di kelas {c}. Suara yang pecah membuat AI sulit mengekstrak pola fitur Mel-Spectrogram.")
                
    print("-" * 50)
print("\nDeep Audit Selesai.")
