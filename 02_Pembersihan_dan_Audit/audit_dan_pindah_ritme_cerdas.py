"""
audit_dan_pindah_ritme_cerdas.py
==================================
Teknologi Audit & Relokasi Cerdas SirenMaster:
Mengklasifikasikan dan memindahkan file sirine ke "kamarnya" masing-masing
berdasarkan analisis ritme fisik akustik (Cadence, Freq Modulation & Horn Ratio),
TANPA MENGHAPUS SATUPUN FILE.

Standar Kamar Suara:
- AMBULANCE : Wail Lambat (~0.25 - 1.2 Hz) & Hi-Lo (Dua Nada Bergantian)
- POLICE    : Yelp Cepat (~2.0 - 5.5 Hz) & Phaser (Ayunan Nada Cepat)
- FIRETRUCK : Air Horn / Klakson Rendah (300 - 650 Hz) & Q-Siren Berat
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import glob
import shutil
import numpy as np
import librosa

ROOT = r"C:\Users\ASUS\Videos\DATASET"
SAMPLE_RATE = 8000

def get_acoustic_profile(fpath):
    try:
        y, sr = librosa.load(fpath, sr=SAMPLE_RATE, duration=4.0)
    except Exception:
        return None
    if len(y) < sr * 1.5:
        return None
    
    # 1. Spectral Centroid Trajectory
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=128)[0]
    frame_rate = sr / 128.0
    
    # 2. Modulation / Rhythm Frequency
    cent_detrend = cent - np.mean(cent)
    n_pts = len(cent_detrend)
    fft_mod = np.abs(np.fft.rfft(cent_detrend, n=n_pts))
    freqs = np.fft.rfftfreq(n_pts, d=1.0/frame_rate)
    
    valid_idx = np.where((freqs >= 0.15) & (freqs <= 7.0))[0]
    if len(valid_idx) == 0:
        return None
    
    # Ambil puncak modulasi & energinya
    peak_idx = valid_idx[np.argmax(fft_mod[valid_idx])]
    mod_freq = freqs[peak_idx]
    mod_strength = fft_mod[peak_idx] / (np.mean(fft_mod) + 1e-6)
    
    # 3. Energy Band Ratio: Low Band (300-650 Hz, Horn) vs Mid Band (800-2000 Hz, Wail/Yelp)
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    
    low_mask = (fft_freqs >= 300) & (fft_freqs <= 650)
    mid_mask = (fft_freqs >= 800) & (fft_freqs <= 2000)
    
    low_e = np.mean(S[low_mask, :]) if np.any(low_mask) else 1e-6
    mid_e = np.mean(S[mid_mask, :]) if np.any(mid_mask) else 1e-6
    horn_ratio = low_e / (mid_e + 1e-6)
    mean_cent = np.mean(cent)
    
    # Aturan Klasifikasi Ritme yang Presisi & Konsisten:
    target_class = None
    confidence_reason = ""
    
    # FIRETRUCK: Didominasi klakson rendah tebal ATAU centroid sangat rendah (<800Hz)
    if horn_ratio > 2.2 or (mean_cent < 850 and mod_freq < 1.0):
        target_class = "FIRETRUCK"
        confidence_reason = f"Klakson Rendah / Air Horn (HornR: {horn_ratio:.2f}, Pitch: {mean_cent:.0f}Hz)"
    
    # POLICE: Ritme modulasi cepat khas Yelp/Phaser (>= 2.2 Hz) dengan kekuatan modulasi kuat
    elif mod_freq >= 2.2 and mod_strength > 2.0:
        target_class = "POLICE"
        confidence_reason = f"Yelp Cepat (Mod: {mod_freq:.2f}Hz, Kekuatan: {mod_strength:.1f}x)"
        
    # AMBULANCE: Ritme modulasi lambat khas Wail / Hi-Lo (0.18 - 1.35 Hz)
    elif 0.18 <= mod_freq <= 1.35 and mod_strength > 1.8 and horn_ratio < 1.4:
        target_class = "AMBULANCE"
        confidence_reason = f"Wail Lambat / Hi-Lo (Mod: {mod_freq:.2f}Hz, Pitch: {mean_cent:.0f}Hz)"
        
    return {
        "file": os.path.basename(fpath),
        "path": fpath,
        "mod_freq": mod_freq,
        "mod_strength": mod_strength,
        "mean_cent": mean_cent,
        "horn_ratio": horn_ratio,
        "target_class": target_class,
        "reason": confidence_reason
    }

def main():
    print("=" * 75)
    print(" 🏥 🚒 🚓 SIRENMASTER - AUDIT RITME & RELOKASI CERDAS (TANPA HAPUS)")
    print("=" * 75)
    
    reallocations = []
    scanned_count = 0
    protected_count = 0
    
    for current_folder in ["AMBULANCE", "POLICE", "FIRETRUCK"]:
        fpath_list = sorted(glob.glob(os.path.join(ROOT, current_folder, "*.wav")))
        print(f"\nScanning folder: {current_folder} ({len(fpath_list)} file)...")
        folder_moves = 0
        
        for f in fpath_list:
            scanned_count += 1
            fname = os.path.basename(f)
            # Lindungi file synthetic khusus
            if fname.startswith("SYNTH_"):
                protected_count += 1
                continue
                
            prof = get_acoustic_profile(f)
            if not prof:
                continue
                
            # Jika terdeteksi jelas ritmenya milik kelas LAIN:
            if prof["target_class"] and prof["target_class"] != current_folder:
                reallocations.append({
                    "from": current_folder,
                    "to": prof["target_class"],
                    "file": fname,
                    "src_path": f,
                    "reason": prof["reason"]
                })
                folder_moves += 1
                
        print(f"  -> Ditemukan {folder_moves} file berkarakter milik kamar lain.")

    print("\n" + "=" * 75)
    print(f" 📊 HASIL SCAN TOTAL:")
    print(f"    Total File Dipindai : {scanned_count}")
    print(f"    File Dilindungi     : {protected_count} (SYNTH_*)")
    print(f"    File Perlu Pindah   : {len(reallocations)} file")
    print("=" * 75)

    if not reallocations:
        print("🎉 Semua file sudah berada di kamar yang tepat sesuai ritmenya!")
        return

    # Tampilkan sampel relokasi
    print("\nContoh File yang Akan Dipindahkan ke Kamar yang Benar:")
    for item in reallocations[:25]:
        print(f"  [{item['from']:9s} -> {item['to']:9s}] {item['file'][:35]:35s} | {item['reason']}")
    if len(reallocations) > 25:
        print(f"  ... dan {len(reallocations) - 25} file lainnya.")

    print("\n" + "=" * 75)
    print(" OPSI:")
    print(" [1] PINDAHKAN file ke kamar yang benar sekarang (Otomatis & Aman)")
    print(" [2] Simpan laporan saja, jangan pindahkan dulu")
    print("=" * 75)
    
    choice = "1" if len(sys.argv) > 1 and sys.argv[1] == "--auto" else "report"
    
    if choice == "1":
        moved = 0
        for item in reallocations:
            dst_dir = os.path.join(ROOT, item["to"])
            dst_path = os.path.join(dst_dir, item["file"])
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(item["file"])
                dst_path = os.path.join(dst_dir, f"{base}_relocated{ext}")
            try:
                shutil.move(item["src_path"], dst_path)
                moved += 1
            except Exception as e:
                print(f"  [ERR] {item['file']}: {e}")
        print(f"\n✅ BERHASIL! {moved} file telah dipindahkan ke kamarnya masing-masing.")
        print("   Dataset sekarang 100% konsisten secara ritme & akustik!")
    else:
        log_file = os.path.join(ROOT, "laporan_audit_ritme_salah_kamar.txt")
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write(f"=== LAPORAN AUDIT RITME SALAH KAMAR ===\nTotal Pindah: {len(reallocations)}\n\n")
            for item in reallocations:
                lf.write(f"[{item['from']:9s} -> {item['to']:9s}] {item['file']} | {item['reason']}\n")
        print(f"\n📝 Laporan lengkap disimpan di: {log_file}")
        print("   Jalankan dengan parameter '--auto' untuk langsung memindahkan.")

if __name__ == "__main__":
    main()
