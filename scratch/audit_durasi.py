import os
import wave

DATASET_BASE = r"C:\Users\ASUS\Videos\DATASET"
CATEGORIES = ['AMBULANCE', 'FIRETRUCK', 'NORMAL', 'POLICE']

print("=" * 65)
print("  AUDIT DURASI SEMUA FILE WAV PER KATEGORI")
print("=" * 65)

for cat in CATEGORIES:
    cat_path = os.path.join(DATASET_BASE, cat)
    if not os.path.exists(cat_path):
        print(f"[!] Folder {cat} tidak ditemukan!")
        continue
    
    files = [f for f in os.listdir(cat_path) if f.endswith('.wav')]
    durations = []
    error_count = 0
    
    for f in files:
        f_path = os.path.join(cat_path, f)
        try:
            with wave.open(f_path, 'r') as w:
                dur = w.getnframes() / float(w.getframerate())
                durations.append(dur)
        except:
            error_count += 1
    
    if len(durations) == 0:
        print(f"\n[{cat}] Tidak ada file valid!")
        continue
    
    # Hitung distribusi durasi
    dur_buckets = {}
    for d in durations:
        bucket = round(d, 1)  # Bulatkan ke 0.1 detik
        dur_buckets[bucket] = dur_buckets.get(bucket, 0) + 1
    
    min_dur = min(durations)
    max_dur = max(durations)
    avg_dur = sum(durations) / len(durations)
    total_dur = sum(durations)
    
    # Hitung berapa yang < 4 detik
    under_4 = sum(1 for d in durations if d < 3.9)
    exactly_4 = sum(1 for d in durations if 3.9 <= d <= 4.1)
    over_4 = sum(1 for d in durations if d > 4.1)
    
    print(f"\n{'='*65}")
    print(f"[{cat}] Total: {len(durations)} file | Error: {error_count}")
    print(f"-" * 65)
    print(f"  Durasi MIN    : {min_dur:.2f} detik")
    print(f"  Durasi MAX    : {max_dur:.2f} detik")
    print(f"  Durasi RATA2  : {avg_dur:.2f} detik")
    print(f"  Total Durasi  : {total_dur/60:.1f} menit ({total_dur/3600:.2f} jam)")
    print(f"-" * 65)
    print(f"  < 4 detik     : {under_4} file ({under_4/len(durations)*100:.1f}%)")
    print(f"  = 4 detik     : {exactly_4} file ({exactly_4/len(durations)*100:.1f}%)")
    print(f"  > 4 detik     : {over_4} file ({over_4/len(durations)*100:.1f}%)")
    
    # Top 5 durasi paling sering muncul
    sorted_buckets = sorted(dur_buckets.items(), key=lambda x: -x[1])[:5]
    print(f"-" * 65)
    print(f"  Top 5 durasi paling sering:")
    for dur_val, count in sorted_buckets:
        bar = "#" * min(int(count / len(durations) * 50), 50)
        print(f"    {dur_val:5.1f} detik : {count:5d} file ({count/len(durations)*100:5.1f}%) {bar}")

print(f"\n{'='*65}")
print("AUDIT SELESAI")
print(f"{'='*65}")
