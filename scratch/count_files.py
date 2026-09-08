import glob
cats = ['POLICE','FIRETRUCK','AMBULANCE','NORMAL']
print("=== Status Dataset Sekarang ===")
for cat in cats:
    n = len(glob.glob(f"{cat}/*.wav"))
    # hitung file baru dari download v3
    new = len(glob.glob(f"{cat}/dl_v3_*.wav"))
    print(f"  {cat}: {n} total  ({new} baru dari download v3)")
