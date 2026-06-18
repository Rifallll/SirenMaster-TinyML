import os
import random
import time
import winsound
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATASET_BASE = r"C:\Users\ASUS\Videos\DATASET"
# KHUSUS SIRINE SAJA (tanpa NORMAL)
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'POLICE']

def main():
    print("=" * 60)
    print("  AUTO-DJ TESTER - KHUSUS SIRINE (TANPA NORMAL)")
    print("=" * 60)
    
    # 1. Siapkan Dataset
    print("[*] Menyiapkan daftar lagu sirine...")
    file_pool = {}
    total_files = 0
    for cat in CATEGORIES:
        path = os.path.join(DATASET_BASE, cat)
        if os.path.exists(path):
            file_pool[cat] = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
            total_files += len(file_pool[cat])
            print(f"  {cat}: {len(file_pool[cat])} file")
            
    print(f"[*] Total file sirine: {total_files}")
    
    # 2. Pengaturan Ujian
    ROUNDS = 100
    PLAY_SECONDS = 10    # Putar 10 detik
    PAUSE_SECONDS = 10   # Jeda 10 detik
    
    print(f"\n[MULAI] {ROUNDS} RONDE | Putar {PLAY_SECONDS}s | Jeda {PAUSE_SECONDS}s")
    print(f"[INFO] Total waktu: ~{ROUNDS * (PLAY_SECONDS + PAUSE_SECONDS) // 60} menit")
    print("Dekatkan mic ESP32 ke speaker laptop!\n")
    time.sleep(3)
    
    for i in range(ROUNDS):
        # Pilih suara sirine acak
        true_cat = random.choice(CATEGORIES)
        true_file = random.choice(file_pool[true_cat])
        file_name = os.path.basename(true_file)
        
        print("=" * 60)
        print(f"[{i+1}/{ROUNDS}] MEMUTAR : {true_cat}")
        print(f"         File : {file_name}")
        print("=" * 60)
        
        # Putar loop selama 10 detik
        winsound.PlaySound(true_file, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
        time.sleep(PLAY_SECONDS)
        winsound.PlaySound(None, winsound.SND_PURGE)
        
        print(f"[JEDA] {PAUSE_SECONDS} detik...")
        time.sleep(PAUSE_SECONDS)
        print("")
        
    print("=" * 60)
    print("  PENGUJIAN SELESAI!")
    print("=" * 60)

if __name__ == "__main__":
    main()
