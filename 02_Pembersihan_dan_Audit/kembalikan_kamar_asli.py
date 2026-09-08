import os
import glob
import shutil

ROOT = r"C:\Users\ASUS\Videos\DATASET"

print("=" * 60)
print("MEMULIHKAN & MENYESUAIKAN FILE KE FOLDER ASLINYA SESUAI NAMA")
print("=" * 60)

moved = 0
for folder in ["AMBULANCE", "FIRETRUCK", "POLICE"]:
    files = glob.glob(os.path.join(ROOT, folder, "*.wav"))
    for fpath in files:
        fname = os.path.basename(fpath).lower()
        target_dir = None
        
        # Tentukan kamar yang tepat berdasarkan nama file aslinya
        if fname.startswith("ambulance_") or fname.startswith("dl_v3_amb_") or fname.startswith("synth_amb_") or "ambulan" in fname:
            target_dir = os.path.join(ROOT, "AMBULANCE")
        elif fname.startswith("fire_") or fname.startswith("dl_v3_dam_") or fname.startswith("synth_fire_") or "damkar" in fname or "fire" in fname:
            target_dir = os.path.join(ROOT, "FIRETRUCK")
        elif fname.startswith("police_") or fname.startswith("dl_v3_pol_") or fname.startswith("synth_pol_") or "polisi" in fname or "police" in fname:
            target_dir = os.path.join(ROOT, "POLICE")
            
        if target_dir and os.path.abspath(target_dir) != os.path.abspath(os.path.dirname(fpath)):
            dst = os.path.join(target_dir, os.path.basename(fpath))
            if os.path.exists(dst):
                # jika nama sama, timpa/ganti
                os.remove(fpath)
            else:
                shutil.move(fpath, dst)
            moved += 1

print(f"Total file dipulihkan dan disesuaikan: {moved}")

for d in ["AMBULANCE", "FIRETRUCK", "POLICE"]:
    all_f = glob.glob(os.path.join(ROOT, d, "*.wav"))
    print(f"📁 {d}: {len(all_f)} file")
