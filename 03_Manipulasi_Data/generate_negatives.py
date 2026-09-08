import os
import numpy as np
import soundfile as sf

dataset_dir = r"c:\Users\ASUS\Videos\DATASET\NORMAL"
SAMPLE_RATE = 8000
DURATION = 2.0  # 2 detik per file
N_SAMPLES = int(SAMPLE_RATE * DURATION)

def generate_brown_noise():
    # Mensimulasikan gemuruh mesin mobil dan suara kabin tertutup
    brown = np.zeros(N_SAMPLES)
    for i in range(1, N_SAMPLES):
        brown[i] = 0.98 * brown[i-1] + np.random.normal(0, 0.1)
    # Hilangkan DC offset
    brown = brown - np.mean(brown)
    return brown / np.max(np.abs(brown))

def generate_turn_signal():
    # Mensimulasikan suara cetek-cetek lampu sein mobil setiap 0.5 detik
    audio = np.zeros(N_SAMPLES)
    interval = int(SAMPLE_RATE * 0.5) # Cetek setiap 0.5s
    for i in range(0, N_SAMPLES, interval):
        # Suara cetek tajam (durasi 30ms)
        t = np.arange(int(SAMPLE_RATE * 0.03)) / SAMPLE_RATE
        env = np.exp(-150 * t)
        # Campuran high frequency click
        click = np.random.normal(0, 1, len(t)) * env
        
        end_idx = min(i+len(click), N_SAMPLES)
        audio[i:end_idx] = click[:end_idx-i]
    return audio / np.max(np.abs(audio))

def generate_babble():
    # Mensimulasikan frekuensi dan ritme orang berbicara (Vokal)
    noise = np.random.normal(0, 1, N_SAMPLES)
    
    # Filter agar hanya frekuensi rendah (vokal manusia) yang dominan
    filtered_noise = np.zeros(N_SAMPLES)
    for i in range(1, N_SAMPLES):
        filtered_noise[i] = 0.8 * filtered_noise[i-1] + 0.2 * noise[i]
        
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    # Amplitudo yang naik turun seperti ritme orang mengobrol (suku kata)
    mod1 = np.sin(2 * np.pi * 3.5 * t) * 0.5 + 0.5  # 3.5 suku kata per detik
    mod2 = np.sin(2 * np.pi * 1.5 * t) * 0.5 + 0.5  # Jeda nafas
    
    babble = filtered_noise * mod1 * mod2
    babble = babble - np.mean(babble)
    return babble / np.max(np.abs(babble))

print("Membuat Suara Sintetis Khusus (Sein Mobil, Kabin Kedap, Orang Ngobrol)...")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

count = 0

# 1. Generate 200 Suara Sein Mobil + Suara Kabin
for i in range(200):
    vol_sein = np.random.uniform(0.5, 1.0)
    vol_kabin = np.random.uniform(0.1, 0.4)
    audio = (generate_turn_signal() * vol_sein) + (generate_brown_noise() * vol_kabin)
    sf.write(os.path.join(dataset_dir, f"synthetic_sein_mobil_{i}.wav"), audio, SAMPLE_RATE)
    count += 1

# 2. Generate 300 Suara Orang Mengobrol (Babble) + Suara Kabin
for i in range(300):
    vol_bicara = np.random.uniform(0.5, 1.0)
    vol_kabin = np.random.uniform(0.1, 0.3)
    audio = (generate_babble() * vol_bicara) + (generate_brown_noise() * vol_kabin)
    sf.write(os.path.join(dataset_dir, f"synthetic_orang_ngobrol_{i}.wav"), audio, SAMPLE_RATE)
    count += 1

# 3. Generate 100 Suara Mesin Mobil Kedap (Rumble)
for i in range(100):
    audio = generate_brown_noise()
    sf.write(os.path.join(dataset_dir, f"synthetic_mesin_kabin_{i}.wav"), audio, SAMPLE_RATE)
    count += 1

print(f"[SELESAI] Berhasil menciptakan {count} file suara negatif buatan ke dalam folder NORMAL!")
