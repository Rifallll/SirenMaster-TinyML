import numpy as np
from scipy.io.wavfile import write
import os

SAMPLE_RATE = 16000
DURATION = 4.0

def generate_wave(freq, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return np.sin(freq * t * 2 * np.pi)

def apply_envelope(audio, attack=0.1, decay=0.1):
    N = len(audio)
    attack_samples = int(attack * SAMPLE_RATE)
    decay_samples = int(decay * SAMPLE_RATE)
    env = np.ones(N)
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
    if decay_samples > 0:
        env[-decay_samples:] = np.linspace(1, 0, decay_samples)
    return audio * env

def generate_ambulance():
    # Hi-Lo: alternates between 900Hz and 1000Hz
    audio = []
    for _ in range(4): # 4 seconds = 4 cycles of 1 sec
        audio.append(generate_wave(900, 0.5))
        audio.append(generate_wave(1000, 0.5))
    return np.concatenate(audio)

def generate_police():
    # Yelp: fast sweep from 700Hz to 1500Hz
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    # LFM sweep
    freqs = 700 + 800 * (np.sin(2 * np.pi * 3 * t) + 1) / 2
    phase = np.cumsum(freqs) / SAMPLE_RATE * 2 * np.pi
    return np.sin(phase)

def generate_firetruck():
    # Wail: slow sweep
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    freqs = 500 + 300 * np.sin(2 * np.pi * 0.25 * t)
    phase = np.cumsum(freqs) / SAMPLE_RATE * 2 * np.pi
    return np.sin(phase)

def add_noise(audio, noise_level=0.1):
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise

def save_wav(filename, audio):
    # Normalize
    audio = audio / np.max(np.abs(audio))
    # Convert to 16-bit PCM
    audio_int16 = np.int16(audio * 32767)
    write(filename, SAMPLE_RATE, audio_int16)

def main():
    os.makedirs("AMBULANCE", exist_ok=True)
    os.makedirs("POLICE", exist_ok=True)
    os.makedirs("FIRETRUCK", exist_ok=True)
    
    print("Generating synthetic real-world sirens...")
    
    for i in range(20):
        # Add varied noise and doppler effects (volume shifts)
        vol_env = np.linspace(0.3, 1.0, int(SAMPLE_RATE * DURATION)) if i % 2 == 0 else np.linspace(1.0, 0.3, int(SAMPLE_RATE * DURATION))
        
        amb = generate_ambulance() * vol_env
        amb = add_noise(amb, noise_level=np.random.uniform(0.05, 0.3))
        save_wav(f"AMBULANCE/SYNTH_amb_{i}.wav", amb)
        
        pol = generate_police() * vol_env
        pol = add_noise(pol, noise_level=np.random.uniform(0.05, 0.3))
        save_wav(f"POLICE/SYNTH_pol_{i}.wav", pol)
        
        fire = generate_firetruck() * vol_env
        fire = add_noise(fire, noise_level=np.random.uniform(0.05, 0.3))
        save_wav(f"FIRETRUCK/SYNTH_fire_{i}.wav", fire)
        
    print("Generated 60 high-quality synthetic sirens! Done.")

if __name__ == "__main__":
    main()
