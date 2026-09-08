import os
import time
import soundfile as sf
try:
    import sounddevice as sd
except ImportError:
    os.system("pip install sounddevice")
    import sounddevice as sd

def rekam_suara(kategori, durasi=5):
    RATE = 8000
    folder = os.path.join(os.path.dirname(__file__), kategori)
    os.makedirs(folder, exist_ok=True)
    
    # Nama file khusus agar diprioritaskan oleh AI (memakai kata 'hard_neg')
    filename = os.path.join(folder, f"hard_neg_live_{int(time.time())}.wav")
    
    print("\n" + "="*50)
    print(f" BERSIAP MEREKAM UNTUK KELAS: {kategori}")
    print("="*50)
    print(" 3...")
    time.sleep(1)
    print(" 2...")
    time.sleep(1)
    print(" 1...")
    time.sleep(1)
    
    print("\n 🔴 MEREKAM SEKARANG! (Silakan siul / teriak / bunyikan suara)")
    rekaman = sd.rec(int(durasi * RATE), samplerate=RATE, channels=1, dtype='float32')
    sd.wait()
    
    sf.write(filename, rekaman, RATE)
    print(f"\n ✅ BERHASIL! File tersimpan di: {filename}")
    print("="*50)

if __name__ == "__main__":
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=========================================")
        print(" PEREKAM DATASET MANUAL (OTOMATIS MASUK)")
        print("=========================================")
        print(" 1. Rekam Suara Bising/Siulan (Masuk NORMAL)")
        print(" 2. Rekam Sirine Ambulans (Masuk AMBULANCE)")
        print(" 3. Rekam Sirine Polisi (Masuk POLICE)")
        print(" 4. Rekam Sirine Damkar (Masuk FIRETRUCK)")
        print(" 0. KELUAR")
        
        pilihan = input("\n Masukkan pilihan (0-4): ").strip()
        
        if pilihan == '1':
            rekam_suara("NORMAL")
            input("\n[Tekan ENTER untuk merekam lagi...]")
        elif pilihan == '2':
            rekam_suara("AMBULANCE")
            input("\n[Tekan ENTER untuk merekam lagi...]")
        elif pilihan == '3':
            rekam_suara("POLICE")
            input("\n[Tekan ENTER untuk merekam lagi...]")
        elif pilihan == '4':
            rekam_suara("FIRETRUCK")
            input("\n[Tekan ENTER untuk merekam lagi...]")
        elif pilihan == '0':
            break
        else:
            print("Pilihan salah!")
            time.sleep(1)
