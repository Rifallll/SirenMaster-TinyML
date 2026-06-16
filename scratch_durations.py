import os
import soundfile as sf
import numpy as np
import pandas as pd

DATASET_PATH = r"C:\Users\ASUS\Videos\DATASET"
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'POLICE', 'NORMAL']

def main():
    print("="*60)
    print("AUDIO PROPERTIES & DURATION ANALYSIS")
    print("="*60)
    
    records = []
    
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATASET_PATH, cat)
        if not os.path.exists(cat_dir):
            continue
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if not file.lower().endswith('.wav'):
                    continue
                file_path = os.path.join(root, file)
                try:
                    info = sf.info(file_path)
                    records.append({
                        'category': cat,
                        'filename': file,
                        'duration': info.duration,
                        'samplerate': info.samplerate,
                        'channels': info.channels
                    })
                except Exception as e:
                    pass
                    
    df = pd.DataFrame(records)
    print(f"Total valid WAV files: {len(df)}")
    
    print("\n[SAMPLE RATE DISTRIBUTION]")
    print(df.groupby(['category', 'samplerate']).size().unstack(fill_value=0))
    
    print("\n[CHANNELS DISTRIBUTION]")
    print(df.groupby(['category', 'channels']).size().unstack(fill_value=0))
    
    print("\n[DURATION STATS (seconds)]")
    stats = df.groupby('category')['duration'].agg(['min', 'mean', 'max', 'std', 'count'])
    print(stats.to_string())
    
    print("\n[SHORT FILES DETAILS (Duration < 1.0s)]")
    short_df = df[df['duration'] < 1.0]
    print(f"Total files shorter than 1.0s: {len(short_df)}")
    if len(short_df) > 0:
        print(short_df.groupby('category').size())
        
    print("\n[LONG FILES DETAILS (Duration > 5.0s)]")
    long_df = df[df['duration'] > 5.0]
    print(f"Total files longer than 5.0s: {len(long_df)}")
    if len(long_df) > 0:
        print(long_df.groupby('category').size())
        
    print("="*60)

if __name__ == '__main__':
    main()
