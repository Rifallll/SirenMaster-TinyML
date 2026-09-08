"""
Audit file POLICE bermasalah secara otomatis.
Cek karakteristik audio: apakah frekuensi dominannya mirip ambulance atau police.
"""
import os
import numpy as np
import librosa

POLICE_DIR = r"C:\Users\ASUS\Videos\DATASET\POLICE"
AMBULANCE_DIR = r"C:\Users\ASUS\Videos\DATASET\AMBULANCE"
SAMPLE_RATE = 8000

# File-file yang dicurigai dari evaluasi
SUSPECT_FILES = [
    "police_0002_seg01.wav",
    "police_0002_seg80.wav",
    "police_0002_seg81.wav",
    "police_0002_seg83.wav",
    "police_0002_seg85.wav",
    "police_0002_seg89.wav",
    "police_0413_original.wav",
    "police_0413_loud.wav",
    "police_0413_noise_light.wav",
    "police_0319_seg01.wav",
    "police_0321_seg01.wav",
    "police_0073_loud.wav",
    "police_0067_shift.wav",
    "police_0163_seg01.wav",
    "police_0103_seg01.wav",
    "Suara Sirine Polisi.wav",
    "Suara Sirine Mobil Patwal Polisi FULL HD.wav",
    "sound_650_1.wav",
    "sound_680_1.wav",
    "sound_758_1.wav",
    "sound_759_1.wav",
]

def analyze_audio(filepath):
    """Analisis karakteristik audio untuk menentukan apakah mirip ambulance atau police."""
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=4.0)
        if len(y) < sr:
            return None
        
        # Hitung spektrum frekuensi
        D = np.abs(librosa.stft(y, n_fft=256, hop_length=128))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=256)
        
        # Cari frekuensi dominan
        mean_spectrum = np.mean(D, axis=1)
        dominant_freq_idx = np.argmax(mean_spectrum)
        dominant_freq = freqs[dominant_freq_idx]
        
        # Estimasi laju modulasi (kecepatan naik-turun nada)
        # Gunakan onset strength sebagai proxy
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if len(onset_env) > 1:
            # FFT dari onset untuk cari laju modulasi
            onset_fft = np.abs(np.fft.rfft(onset_env))
            onset_freqs = np.fft.rfftfreq(len(onset_env), d=128/sr)
            # Frekuensi modulasi dominan (Hz)
            mod_freq_idx = np.argmax(onset_fft[1:]) + 1  # skip DC
            mod_freq = onset_freqs[mod_freq_idx]
        else:
            mod_freq = 0
        
        # RMS (volume rata-rata)
        rms = np.sqrt(np.mean(y**2))
        
        # Spektral centroid (tinggi/rendahnya nada rata-rata)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        return {
            'dominant_freq': dominant_freq,
            'mod_freq': mod_freq,
            'rms': rms,
            'centroid': centroid,
            'duration': len(y)/sr
        }
    except Exception as e:
        return None

print("=" * 70)
print("AUDIT FILE POLICE BERMASALAH")
print("=" * 70)
print(f"{'File':<45} {'Dom.Freq':>9} {'Mod.Rate':>9} {'Centroid':>9} {'Status'}")
print("-" * 70)

# Ambil referensi karakteristik ambulance
print("\n[Menghitung referensi AMBULANCE...]")
ambu_centroids = []
ambu_files = [f for f in os.listdir(AMBULANCE_DIR) if f.endswith('.wav') and 'aug' not in f][:10]
for af in ambu_files:
    res = analyze_audio(os.path.join(AMBULANCE_DIR, af))
    if res:
        ambu_centroids.append(res['centroid'])
ambu_centroid_avg = np.mean(ambu_centroids) if ambu_centroids else 800
print(f"  Rata-rata centroid AMBULANCE: {ambu_centroid_avg:.0f} Hz")

# Ambil referensi police yang benar
print("[Menghitung referensi POLICE yang benar...]")
police_ok_files = [f for f in os.listdir(POLICE_DIR) if f.endswith('.wav') and 'aug' not in f 
                   and not any(s in f for s in ['0002', '0413', '0319', '0321', '0073', '0067', '0163', '0103'])][:10]
police_centroids = []
for pf in police_ok_files:
    res = analyze_audio(os.path.join(POLICE_DIR, pf))
    if res:
        police_centroids.append(res['centroid'])
police_centroid_avg = np.mean(police_centroids) if police_centroids else 1200
print(f"  Rata-rata centroid POLICE (normal): {police_centroid_avg:.0f} Hz")

print()
print(f"{'File':<45} {'Dom.Freq':>9} {'Centroid':>9} {'Mirip?'}")
print("-" * 70)

problematic = []
for fname in SUSPECT_FILES:
    fpath = os.path.join(POLICE_DIR, fname)
    if not os.path.exists(fpath):
        # Coba cari file yang namanya mengandung bagian tersebut
        all_files = os.listdir(POLICE_DIR)
        matches = [f for f in all_files if fname.replace('.wav','') in f]
        if matches:
            fpath = os.path.join(POLICE_DIR, matches[0])
            fname = matches[0]
        else:
            print(f"  {'[TIDAK ADA]':<45}")
            continue
    
    res = analyze_audio(fpath)
    if res is None:
        print(f"  {fname:<45} [GAGAL BACA]")
        continue
    
    # Tentukan mirip ambulance atau police berdasarkan centroid
    diff_ambu = abs(res['centroid'] - ambu_centroid_avg)
    diff_police = abs(res['centroid'] - police_centroid_avg)
    
    if diff_ambu < diff_police:
        status = "[!] MIRIP AMBULANCE"
        problematic.append(fpath)
    else:
        status = "[OK] Police"
    
    print(f"  {fname:<45} {res['dominant_freq']:>8.0f}Hz {res['centroid']:>8.0f}Hz  {status}")

print()
print("=" * 70)
print(f"TOTAL FILE BERMASALAH (mirip ambulance): {len(problematic)}")
print()

# Simpan daftar file bermasalah
with open("problematic_police_files.txt", "w") as f:
    for p in problematic:
        f.write(p + "\n")
print("Daftar tersimpan di: problematic_police_files.txt")
