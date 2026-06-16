import numpy as np
import os

CACHE_PATH = "siren_40x249_melspec_cache_lokal.npz"
SCALER_PATH = "siren_scaler.npz"
REPLAY_PATH = "siren_replay_buffer.npz"

print(f"Loading large cache file: {CACHE_PATH} (please wait, this can take a minute)...")
with np.load(CACHE_PATH, allow_pickle=True) as data:
    X_train = data['X_train']
    y_train = data['y_train']
    
print("Successfully loaded. Computing mean and std...")
global_mean = np.mean(X_train, axis=(0, 1))
global_std = np.std(X_train, axis=(0, 1))
global_std[global_std < 1e-6] = 1.0

# Save scaler values
np.savez(SCALER_PATH, global_mean=global_mean, global_std=global_std)
print(f"Saved global mean and std to {SCALER_PATH}")

# Extract replay buffer (1000 random samples)
print("Extracting 1000 random samples for replay buffer...")
num_samples = len(X_train)
replay_size = min(1000, num_samples)
indices = np.random.choice(num_samples, replay_size, replace=False)

X_replay = X_train[indices]
y_replay = y_train[indices]

np.savez_compressed(REPLAY_PATH, X_replay=X_replay, y_replay=y_replay)
print(f"Saved replay buffer ({replay_size} samples) to {REPLAY_PATH}")
print("Optimization complete!")
