import os
import librosa
import soundfile as sf
import numpy as np

input_dir = r"C:\Users\ASUS\Videos\DATASET\Ambilance"
output_dir = r"C:\Users\ASUS\Videos\DATASET\AMBULANCE"

sr = 8000
chunk_duration = 1.024
samples_per_chunk = int(sr * chunk_duration)
hop_length = samples_per_chunk // 2

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = [f for f in os.listdir(input_dir) if f.endswith('.m4a')]
total_chunks = 0

for file in files:
    try:
        file_path = os.path.join(input_dir, file)
        print(f"Processing {file}...")
        
        # Load audio, resample to 8000 Hz
        y, _ = librosa.load(file_path, sr=sr)
        
        file_base = os.path.splitext(file)[0].replace(" ", "_").lower()
        
        # Split into chunks
        chunk_idx = 0
        for start in range(0, len(y) - samples_per_chunk + 1, hop_length):
            chunk = y[start:start + samples_per_chunk]
            
            # Save chunk
            out_name = f"newamb_{file_base}_{chunk_idx:04d}.wav"
            out_path = os.path.join(output_dir, out_name)
            sf.write(out_path, chunk, sr)
            
            chunk_idx += 1
            total_chunks += 1
            
        print(f"  Created {chunk_idx} chunks from {file}")
            
    except Exception as e:
        print(f"Error processing {file}: {e}")

print(f"\nSuccessfully created {total_chunks} new WAV files in AMBULANCE folder.")
