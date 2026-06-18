import numpy as np

try:
    with np.load("siren_40x249_melspec_cache_lokal.npz", allow_pickle=True) as data:
        print("Keys in npz:", list(data.keys()))
        if 'fingerprint' in data:
            print("Fingerprint:", data['fingerprint'])
        if 'X_train' in data:
            print("X_train shape:", data['X_train'].shape)
            print("y_train shape:", data['y_train'].shape)
            print("y_train unique values:", np.unique(data['y_train'], return_counts=True))
        if 'X_test' in data:
            print("X_test shape:", data['X_test'].shape)
            print("y_test shape:", data['y_test'].shape)
            print("y_test unique values:", np.unique(data['y_test'], return_counts=True))
except Exception as e:
    print("Error loading cache:", e)
