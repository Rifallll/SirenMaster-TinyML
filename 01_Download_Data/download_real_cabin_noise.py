"""
download_real_cabin_noise.py
============================
Script otomatis untuk mengunduh suara asli (REAL) lingkungan dalam mobil, jalan raya,
lalu lintas, obrolan penumpang, batuk, bersin, suara main HP, knalpot brong, tangisan bayi,
palang kereta, alarm mobil, bor aspal proyek, TOA masjid, petir badai, serangga tonggeret,
dan lubang jalan dari YouTube ke folder NORMAL tanpa filter!
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import subprocess
import glob
import librosa
import soundfile as sf
import numpy as np

# Daftar pencarian YouTube untuk suara REAL TAHAP AKHIR (Proyek, TOA, Petir, Serangga, Benturan)
URLS = {
    # 1. Proyek & Perbaikan Jalan
    "bor_aspal_jackhammer": "ytsearch1:suara bor aspal jalan raya jackhammer construction sound effect",
    "alat_berat_excavator": "ytsearch1:suara mesin excavator alat berat proyek jalanan indonesia",
    
    # 2. TOA & Sound System Luar
    "toa_masjid_jalan": "ytsearch1:suara toa masjid keras dari jalan raya pengajian adzan",
    "sound_system_jalanan": "ytsearch1:suara sound system karnaval pawai jalanan indonesia",
    
    # 3. Cuaca Ekstrem & Angin
    "petir_badai_mobil": "ytsearch1:suara petir menggelegar hujan badai thunder sound effect",
    "angin_kencang_tol": "ytsearch1:suara angin kencang di mobil jalan tol wind buffeting sound",
    
    # 4. Benturan Fisik Mobil
    "hantaman_lubang_jalan": "ytsearch1:suara mobil kena lubang jalan polisi tidur pothole sound effect",
    "pintu_mobil_ditutup": "ytsearch1:suara pintu mobil ditutup keras car door slam sound effect",
    
    # 5. Alam Melengking & Hewan
    "serangga_tonggeret": "ytsearch1:suara serangga tonggeret cicada malam hari suara keras melengking",
    "anjing_menggonggong": "ytsearch1:suara anjing menggonggong keras malam hari dog barking sound effect"
}

OUT_DIR = r"c:\Users\ASUS\Videos\DATASET\NORMAL"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("  MENGUNDUH SUARA REAL TAHAP AKHIR (PROYEK, TOA, PETIR, SERANGGA) KE 'NORMAL'")
print("=" * 70)

for name, url in URLS.items():
    print(f"\n[>] Mengunduh topik: {name} ...")
    out_template = os.path.join(OUT_DIR, f"real_ultimate_{name}_%(id)s.%(ext)s")
    
    # Mengambil 3 menit pertama dari video apapun
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "0",
        "--download-sections", "*00:00:00-00:03:00",
        "--no-playlist",
        "-o", out_template,
        url
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[!] Gagal mengunduh {name}: {e}")

print("\n[*] Memotong (slicing) audio menjadi potongan 4 detik...")
mp3_files = glob.glob(os.path.join(OUT_DIR, "real_ultimate_*.mp3")) + glob.glob(os.path.join(OUT_DIR, "real_ultimate_*.wav"))

total_chunks = 0
for wf in mp3_files:
    if "_part" in wf:
        continue
    print(f"  -> Memproses: {os.path.basename(wf)} ...")
    try:
        y, sr = librosa.load(wf, sr=8000)
        chunk_len = int(4.0 * sr)  # 4 detik
        
        saved = 0
        for i in range(0, len(y) - chunk_len, chunk_len):
            chunk = y[i:i+chunk_len]
            rms = np.sqrt(np.mean(chunk**2))
            if rms < 0.001:
                continue
            
            out_name = os.path.splitext(wf)[0] + f"_part{saved:04d}.wav"
            sf.write(out_name, chunk, sr)
            saved += 1
            total_chunks += saved
            
        print(f"     [OK] Menghasilkan {saved} sampel 4-detik.")
        if os.path.exists(wf):
            os.remove(wf)  # Hapus file panjang setelah dipotong
    except Exception as e:
        print(f"     [!] Error memotong {wf}: {e}")

# Bersihkan sisa file mp3 sementara jika ada
for tmp_mp3 in glob.glob(os.path.join(OUT_DIR, "real_ultimate_*.mp3")):
    try:
        os.remove(tmp_mp3)
    except:
        pass

print("\n" + "=" * 70)
print(f"[SUCCESS] Berhasil menambahkan {total_chunks} sampel real tahap akhir ke folder NORMAL!")
print("Sekarang Anda bisa langsung melatih ulang AI dengan perintah:")
print("  python 04_Training_AI\\train_v2_fix_police.py")
print("=" * 70)
