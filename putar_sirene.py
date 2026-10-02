"""
putar_sirene.py
===============
Alat Pemutar Suara Sirene JERNIH & LOUD untuk Pengujian Mandiri di Terminal Anda.
Telah disesuaikan: Menggunakan 100% file rekaman asli murni (tanpa intro/dl_v3/noise).
Setiap suara diputar 4 KALI BERTURUT-TURUT (~16 detik).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os, time, winsound, random, collections, subprocess
from export_benchmark import export_benchmark_to_excel_and_charts, get_next_pengujian_name, OUTPUT_DIR

ROOT = r"C:\Users\ASUS\Videos\DATASET"

# Daftar file tunggal paling JERNIH & BERKUALITAS TINGGI
SINGLE_SAMPLES = {
    "1a": ("AMBULANCE (Wail Murni)", os.path.join(ROOT, "AMBULANCE", "ambulance_0004_seg01.wav")),
    "1b": ("DAMKAR / FIRETRUCK (Q-Siren Murni)", os.path.join(ROOT, "FIRETRUCK", "fire_0002_seg01.wav")),
    "1c": ("POLISI (Yelp Murni)", os.path.join(ROOT, "POLICE", "police_0001_seg05.wav")),
}

def build_exam_10():
    """Pilih 10 suara ACAK dari seluruh dataset setiap kali dipanggil.
    Selalu seimbang: 3 Ambulans + 3 Damkar + 4 Polisi (atau sesuai yang tersedia).
    Hasilnya berbeda setiap kali Menu 2 dibuka!"""
    def get_random_files(folder_name, label, n):
        folder_path = os.path.join(ROOT, folder_name)
        if not os.path.exists(folder_path):
            return []
        prefix_map = {
            "AMBULANCE": "ambulance_",
            "FIRETRUCK": "fire_",
            "POLICE":    "police_"
        }
        pref = prefix_map.get(folder_name, "")
        files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".wav")
            and f.lower().startswith(pref)
            and not f.startswith("aug_")
            and not f.startswith("SYNTH_")
            and "noise" not in f.lower()
        ]
        if not files:
            # Fallback jika belum ada prefix standar
            files = [
                f for f in os.listdir(folder_path)
                if f.endswith(".wav")
                and not f.startswith("aug_")
                and not f.startswith("SYNTH_")
                and "noise" not in f.lower()
            ]
        if not files:
            return []
        picked = random.sample(files, min(n, len(files)))
        return [(f"{label} #{i+1} ({f[:30]})", os.path.join(folder_path, f)) for i, f in enumerate(picked)]

    pool  = get_random_files("AMBULANCE", "AMBULANCE", 3)
    pool += get_random_files("FIRETRUCK",  "DAMKAR",   3)
    pool += get_random_files("POLICE",     "POLISI",   4)
    random.shuffle(pool)
    return pool

# Dipanggil sekali di awal (fallback)
EXAM_10_SAMPLES = build_exam_10()

# Daftar 15 Suara Murni Unik untuk Ujian Random (Menu 3)
RANDOM_POOL_15 = EXAM_10_SAMPLES + [
    ("AMBULANCE #4 (Wail Murni 4)", os.path.join(ROOT, "AMBULANCE", "ambulance_0007_seg01.wav")),
    ("AMBULANCE #5 (Wail Murni 5)", os.path.join(ROOT, "AMBULANCE", "ambulance_0008_seg01.wav")),
    ("DAMKAR #5 (Raungan Murni 5)", os.path.join(ROOT, "FIRETRUCK", "fire_0002_seg05.wav")),
    ("POLISI #4 (Hi-Lo Murni 4)", os.path.join(ROOT, "POLICE", "police_0001_seg08.wav")),
    ("POLICE #5 (Yelp Murni 5)", os.path.join(ROOT, "POLICE", "police_0001_seg09.wav")),
]


def load_audit_blacklist():
    """
    Membaca laporan_audit_ritme_salah_kamar.txt dan mengembalikan
    dict {folder_name: set(filenames)} yang PERLU DIEXCLUDE dari pengujian.
    File yang dianggap salah kamar oleh audit TIDAK akan dimasukkan ke pool pengujian.
    """
    import re as _re
    audit_file = os.path.join(ROOT, "laporan_audit_ritme_salah_kamar.txt")
    blacklist = {"AMBULANCE": set(), "FIRETRUCK": set(), "POLICE": set()}

    if not os.path.exists(audit_file):
        return blacklist  # Tidak ada audit → tidak ada exclusion

    try:
        with open(audit_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return blacklist

    for line in lines:
        line = line.strip()
        # Format: [SRC -> DST] filename.wav | alasan
        m = _re.match(r'\[(AMBULANCE|FIRETRUCK|POLICE)\s*->\s*(AMBULANCE|FIRETRUCK|POLICE)\s*\]\s+(\S+\.wav)', line)
        if m:
            src_folder = m.group(1)
            fname = m.group(3)
            # Hanya exclude _seg01.wav asli (non-aug, non-SYNTH) yang benar-benar ada di disk
            if (fname.endswith("_seg01.wav")
                    and not fname.startswith("aug_")
                    and not fname.startswith("SYNTH_")):
                fpath = os.path.join(ROOT, src_folder, fname)
                if os.path.exists(fpath):
                    blacklist[src_folder].add(fname)

    return blacklist


# Cache blacklist audit (dibaca sekali saat startup, bisa di-reload)
_AUDIT_BLACKLIST = None


def get_audit_blacklist(force_reload=False):
    """Ambil audit blacklist (cached). Set force_reload=True untuk reload dari disk."""
    global _AUDIT_BLACKLIST
    if _AUDIT_BLACKLIST is None or force_reload:
        _AUDIT_BLACKLIST = load_audit_blacklist()
        total_excluded = sum(len(v) for v in _AUDIT_BLACKLIST.values())
        if total_excluded > 0:
            print(f"  [Audit] Blacklist dimuat: {total_excluded} file mislabeled akan dilewati.")
    return _AUDIT_BLACKLIST


def build_pool_100(target_total=100):
    """Bangun sampel uji seimbang per kelas (default 100 suara).
    Menggunakan AUDIT BLACKLIST dari laporan_audit_ritme_salah_kamar.txt.
    Urutan DIACAK secara unik agar pengujian real seperti kondisi nyata.
    Seed = waktu sekarang (setiap run berbeda)."""

    audit_bl = get_audit_blacklist()

    def get_seg_files(folder_name, cls_label, target_n):
        """Ambil file .wav murni tanpa duplikat jika memungkinkan.
        Prioritas 1: File _seg01.wav yang TIDAK ada di audit blacklist (paling bersih).
        Prioritas 2: Jika terlalu sedikit, ambil file _seg01.wav apapun yang tidak di-blacklist.
        Prioritas 3: Jika masih kurang, fallback ke semua .wav non-aug/SYNTH.
        """
        folder_path = os.path.join(ROOT, folder_name)
        if not os.path.exists(folder_path):
            return []
        prefix_map = {
            "AMBULANCE": "ambulance_",
            "FIRETRUCK": "fire_",
            "POLICE":    "police_"
        }

        # Gabungkan: blacklist hardcoded + blacklist dari audit
        HARDCODED_EXCLUDE = {
            "fire_0017_seg01.wav", "fire_0003_seg01.wav", "fire_0093_seg01.wav",
            "fire_0094_seg01.wav", "fire_0095_seg01.wav", "fire_0039_seg01.wav",
            "ambulance_0134_seg01.wav", "ambulance_0146_seg01.wav", "ambulance_0167_seg01.wav",
            "ambulance_0192_seg01.wav", "police_0108_seg01.wav"
        }
        # Ambil blacklist dari audit untuk folder ini
        audit_exclude = audit_bl.get(folder_name, set())
        ALL_EXCLUDE = HARDCODED_EXCLUDE | audit_exclude

        pref = prefix_map.get(folder_name, "")

        # TIER 1: file _seg01.wav AMAN (tidak di blacklist audit)
        files_tier1 = [
            f for f in os.listdir(folder_path)
            if f.endswith("_seg01.wav")
            and f.lower().startswith(pref)
            and not f.startswith("aug_")
            and not f.startswith("SYNTH_")
            and "noise" not in f
            and "shift" not in f
            and "loud"  not in f
            and f not in ALL_EXCLUDE
        ]

        if len(files_tier1) >= target_n:
            # Cukup file tier 1 → pakai semua dari tier1
            picked = random.sample(files_tier1, target_n)
            return [(cls_label, os.path.join(folder_path, f)) for f in picked]

        # Jika tier1 tidak cukup: tampilkan peringatan tapi tetap pakai semua tier1
        # lalu tambahkan dari tier1 dengan repeat (jika sangat sedikit)
        if files_tier1:
            # Repeat dari files yang ada jika tidak cukup
            picked = files_tier1[:]
            while len(picked) < target_n:
                picked += files_tier1
            picked = random.sample(picked[:target_n + len(files_tier1)], target_n)
            return [(cls_label, os.path.join(folder_path, f)) for f in picked[:target_n]]

        # TIER 2: fallback ke _seg01.wav tanpa audit filter (jika tier1 kosong)
        files_tier2 = [
            f for f in os.listdir(folder_path)
            if f.endswith("_seg01.wav")
            and f.lower().startswith(pref)
            and not f.startswith("aug_")
            and not f.startswith("SYNTH_")
            and f not in HARDCODED_EXCLUDE
        ]
        if files_tier2:
            if len(files_tier2) >= target_n:
                picked = random.sample(files_tier2, target_n)
            else:
                picked = (files_tier2 * (target_n // len(files_tier2) + 1))[:target_n]
            return [(cls_label, os.path.join(folder_path, f)) for f in picked]

        # TIER 3: fallback ke semua .wav
        files_tier3 = [
            f for f in os.listdir(folder_path)
            if f.endswith(".wav")
            and f.lower().startswith(pref)
            and not f.startswith("aug_")
            and not f.startswith("SYNTH_")
            and f not in HARDCODED_EXCLUDE
        ]
        if not files_tier3:
            return []
        if len(files_tier3) >= target_n:
            picked = random.sample(files_tier3, target_n)
        else:
            picked = (files_tier3 * (target_n // len(files_tier3) + 1))[:target_n]
        return [(cls_label, os.path.join(folder_path, f)) for f in picked]

    n_amb = target_total // 3
    n_dam = target_total // 3
    n_pol = target_total - (n_amb + n_dam)

    # Tampilkan info dataset sebelum disampling
    bl = audit_bl
    for folder, pref_key, n_target, lbl in [
        ("AMBULANCE", "ambulance_", n_amb, "Ambulance"),
        ("FIRETRUCK",  "fire_",      n_dam, "Damkar"),
        ("POLICE",     "police_",    n_pol, "Polisi"),
    ]:
        folder_path = os.path.join(ROOT, folder)
        if os.path.exists(folder_path):
            all_seg = [f for f in os.listdir(folder_path)
                       if f.endswith("_seg01.wav") and f.lower().startswith(pref_key)
                       and not f.startswith("aug_") and not f.startswith("SYNTH_")]
            n_safe = len([f for f in all_seg if f not in bl.get(folder, set())])
            print(f"  [{lbl}] {n_safe} file aman dari {len(all_seg)} total _seg01.wav (dibutuhkan {n_target})")

    pool  = get_seg_files("AMBULANCE", "AMBULANCE", n_amb)
    pool += get_seg_files("FIRETRUCK", "DAMKAR",    n_dam)
    pool += get_seg_files("POLICE",    "POLISI",    n_pol)

    # Acak MURNI setiap run (berbeda tiap kali) agar pengujian terasa nyata
    random.shuffle(pool)
    return pool



def play_sound(cls_name, filepath, loops=4):
    if not filepath or not os.path.exists(filepath):
        print(f"[!] Error: File audio untuk {cls_name} tidak ditemukan! ({filepath})")
        return
    print("\n" + "="*65)
    print(f" Menguji kelas : {cls_name}")
    print(f" File Audio    : {os.path.basename(filepath)}")
    print(f" Pemutaran     : {loops}x (~{loops*4} Detik HD)")
    print("="*65)
    print(" MEMBUNYIKAN SEKARANG... (Dekatkan alat ke speaker)")
    for i in range(1, loops + 1):
        print(f"    -> Putaran {i} dari {loops}...")
        winsound.PlaySound(filepath, winsound.SND_FILENAME)
        time.sleep(0.1)
    print("\n [+] SELESAI DIPUTAR!")
    print(" -> Perhatikan layar LCD alat Anda (Status mengunci jika >= 50%).")
    print("="*65)


def menu_ujian_100():
    """Menu [4]: Benchmark 100 Sirine dengan Statistik Lengkap, Ekspor Excel & Grafik Resmi Skripsi."""
    KELAS = ["AMBULANCE", "DAMKAR", "POLISI"]
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*72)
    print("  BENCHMARK PENGUJIAN HARDWARE SIRENMASTER (UJIAN SKRIPSI)")
    print("  Dilengkapi Rekam Hasil Manual, Ekspor Excel (.xlsx) & Grafik (.png)")
    print("="*72)
    print("  Pilih jumlah suara yang ingin diuji:")
    print("   [1] 100 Suara (Standar Pengujian Skripsi Penuh)")
    print("   [2] 50 Suara  (Pengujian Sedang)")
    print("   [3] 20 Suara  (Uji Cepat / Cek Alat)")
    print("   [4] Kustom (Tentukan sendiri jumlah suara)")
    print("   [0] Batal / Kembali ke Menu Utama")
    print("="*72)

    try:
        pilih_jml = input("  Masukkan pilihan [1-4 atau tekan ENTER untuk 100 suara]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  [!] Dibatalkan. Kembali ke menu utama.")
        return
    if pilih_jml == "0":
        return
    elif pilih_jml == "2":
        target_jml = 50
    elif pilih_jml == "3":
        target_jml = 20
    elif pilih_jml == "4":
        try:
            target_jml = int(input("  Masukkan jumlah suara: ").strip())
            if target_jml <= 0:
                target_jml = 100
        except (ValueError, EOFError, KeyboardInterrupt):
            target_jml = 100
    else:
        target_jml = 100

    try:
        pilih_loop = input("  Jumlah putaran suara per tes [Tekan ENTER untuk 4x putar (~16s), atau ketik 5]: ").strip()
    except (EOFError, KeyboardInterrupt):
        pilih_loop = ""
    loops_per_sample = 5 if pilih_loop == "5" else 4

    try:
        pool = build_pool_100(target_jml)
    except KeyboardInterrupt:
        print("\n\n[!] Dibatalkan. Kembali ke menu utama.")
        return

    if not pool:
        print("\n[!] POOL SUARA KOSONG.")
        print("    Pastikan folder AMBULANCE, FIRETRUCK, POLICE berisi file .wav.")
        try:
            input("\n[Tekan ENTER untuk kembali ke menu utama...]")
        except (EOFError, KeyboardInterrupt):
            pass
        return

    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*72)
    print(f"  MEMULAI PENGUJIAN {len(pool)} SUARA SIRINE SECARA MANUAL")
    print("="*72)

    # Info dataset yang digunakan
    from collections import Counter as _Ctr
    dist_pool = _Ctr(kls for kls, _ in pool)
    unique_pool = _Ctr(os.path.basename(fpath) for _, fpath in pool)
    total_unique = len(unique_pool)
    bl = get_audit_blacklist()
    bl_total = sum(len(v) for v in bl.values())
    print(f"  Total Sampel : {len(pool)} suara ({total_unique} file unik, {len(pool)-total_unique} file diulang)")
    print(f"  Distribusi   : Ambulance={dist_pool.get('AMBULANCE',0)} | Damkar={dist_pool.get('DAMKAR',0)} | Polisi={dist_pool.get('POLISI',0)}")
    print(f"  Filter Audit : {bl_total} file mislabeled/salah-kamar telah DIKECUALIKAN dari pool")
    print(f"  Pemutaran    : {loops_per_sample}x Putaran (~{loops_per_sample*4} Detik per suara)")
    print(f"  Urutan       : Diacak secara real-time")
    # Peringatan jika ada kelas dengan file sedikit (harus diulang)
    for kls_label, folder in [("AMBULANCE","AMBULANCE"), ("DAMKAR","FIRETRUCK"), ("POLISI","POLICE")]:
        pool_kls = [os.path.basename(fpath) for kl, fpath in pool if kl == kls_label]
        unique_kls = len(set(pool_kls))
        total_kls = len(pool_kls)
        if unique_kls < total_kls:
            print(f"  [!] PERHATIAN: {kls_label} hanya {unique_kls} file aman → {total_kls - unique_kls} file diulang (dataset terbatas)")
    print("-" * 72)
    print("  PANDUAN PENGUJIAN:")
    print(f"  1. Suara diputar {loops_per_sample}x (~{loops_per_sample*4} detik) agar alat punya waktu mengunci.")
    print("  2. Dekatkan mikrofon alat ESP32 SirenMaster ke speaker komputer.")
    print("  3. Perhatikan apa yang tertampil di layar LCD alat Anda.")
    print("  4. Masukkan input manual sesuai tampilan LCD:")
    print("     [a] = Tampil AMBULANCE")
    print("     [d] = Tampil DAMKAR")
    print("     [p] = Tampil POLISI")
    print("     [0] = DIAM / Tidak Bereaksi (System Aman)")

    print("     [l] = LAMBAT (>10 detik tapi benar)")
    print("     [x] = SKIP ronde ini")
    print("="*72)
    try:
        input(f"\n[Tekan ENTER untuk mulai Ronde 1 dari {len(pool)}...]")
    except (EOFError, KeyboardInterrupt):
        print("\n\n  [!] Dibatalkan sebelum pengujian dimulai.")
        return

    confusion = {k: collections.Counter() for k in KELAS + ["TDK_DETEKSI"]}
    mode_count = {k: collections.Counter() for k in KELAS}
    results_log = []
    retrain_list = []
    start_time = time.time()

    try:
        for idx, (kelas_asli, fpath) in enumerate(pool, 1):
            os.system('cls' if os.name == 'nt' else 'clear')
            fname = os.path.basename(fpath)
            
            # Hitung skor berjalan sejauh ini
            done_so_far = len(results_log)
            benar_so_far = sum(1 for r in results_log if r.get("mode") in ("BENAR", "LAMBAT"))
            acc_now = (benar_so_far / done_so_far * 100) if done_so_far > 0 else 0

            print("="*72)
            print(f"  RONDE {idx:03d} / {len(pool):03d}  |  File: {fname[:40]}")
            print(f"  SKOR BERJALAN: {benar_so_far}/{done_so_far} ({acc_now:.1f}%) | Sisa: {len(pool) - idx + 1} Ronde")
            print("="*72)
            print(f"  [*] Memutar suara {loops_per_sample}x... (Dekatkan mic ESP32 ke speaker!)")

            file_ok = os.path.exists(fpath)
            if not file_ok:
                print(f"  [!] File audio tidak ditemukan — Ronde {idx} dilewati.")
                results_log.append({
                    "ronde": idx, "waktu": time.strftime("%H:%M:%S"), "file": fname,
                    "kelas_asli": kelas_asli, "tebakan": "-", "mode": "FILE_MISSING",
                    "skor": 0, "keterangan": "File tidak ditemukan di disk"
                })
                time.sleep(0.5)
                continue

            # Putar audio 4x atau 5x sesuai pilihan
            for lp in range(1, loops_per_sample + 1):
                print(f"      -> Putaran {lp}/{loops_per_sample}... (Dekatkan mic ESP32 ke speaker)", end="\r", flush=True)
                winsound.PlaySound(fpath, winsound.SND_FILENAME)
                time.sleep(0.1)
            print(f"      -> Selesai diputar {loops_per_sample}x (~{loops_per_sample*4} Detik). Perhatikan LCD alat Anda!      ")

            cls_icon = {"AMBULANCE": "🚑 AMBULANCE", "DAMKAR": "🚒 DAMKAR", "POLISI": "🚓 POLISI"}
            print("-" * 72)
            print(f"  TARGET SEBENARNYA:  {cls_icon.get(kelas_asli, kelas_asli)}")
            print("-" * 72)
            print("  APA YANG MUNCUL DI LAYAR LCD ALAT ESP32 ANDA?")
            print("   [a] AMBULANCE")
            print("   [d] DAMKAR")
            print("   [p] POLISI")
            print("   [0] DIAM / TIDAK MERESPON (Masih SYSTEM AMAN)")
            print("   [l] LAMBAT (>10 detik tapi akhirnya benar)")
            print("   [x] SKIP ronde ini")
            print("=" * 72)

            while True:
                ans = input("  Ketik pilihan Anda (a / d / p / 0 / l / x): ").strip().lower()
                if ans in ('a', 'd', 'p', '0', 'l', 'x'):
                    break
                print("  [!] Input tidak valid! Harap ketik a, d, p, 0, l, atau x.")

            map_lcd = {'a': 'AMBULANCE', 'd': 'DAMKAR', 'p': 'POLISI'}
            waktu_uji = time.strftime("%H:%M:%S")

            if ans == 'x':
                tebakan = "-"
                mode = "SKIP"
                skor = 0
                ket = "Dilewati oleh pengguna"
                print("\n  [-] Ronde ini dilewati (SKIP).")

            elif ans == '0':
                tebakan = "TDK_DETEKSI"
                mode = "TIDAK_TERDETEKSI"
                skor = 0
                ket = "LCD diam / tidak merespon (Missed)"
                confusion[kelas_asli]["TDK_DETEKSI"] += 1
                mode_count[kelas_asli]["TIDAK_TERDETEKSI"] += 1
                retrain_list.append((kelas_asli, fpath, "TIDAK TERDETEKSI", "Layar diam / Sistem Aman"))

                print("\n" + "~"*72)
                print("  STATUS: [-] TIDAK TERDETEKSI / MISSED (Skor 0)")
                print(f"  Suara Asli : {kelas_asli}")
                print("  LCD Tampil : SYSTEM AMAN / DIAM (Tidak bereaksi)")
                print("  -> Data tersimpan ke Excel sebagai TIDAK TERDETEKSI.")
                print("~"*72)

            elif ans == 'l':
                tebakan = kelas_asli
                mode = "LAMBAT"
                skor = 1
                ket = "Terdeteksi tepat namun butuh waktu >10 detik"
                confusion[kelas_asli][kelas_asli] += 1
                mode_count[kelas_asli]["LAMBAT"] += 1

                print("\n" + "="*72)
                print("  STATUS: [~] BENAR TAPI LAMBAT (Skor 1)")
                print(f"  Suara Asli : {kelas_asli}")
                print(f"  LCD Tampil : {kelas_asli} (Terkunci >10 detik)")
                print("  -> Dihitung BENAR, tercatat di Excel sebagai LAMBAT.")
                print("="*72)

            else:
                tebakan = map_lcd[ans]
                if tebakan == kelas_asli:
                    mode = "BENAR"
                    skor = 1
                    ket = "Tepat sesuai suara asli"
                    confusion[kelas_asli][kelas_asli] += 1
                    mode_count[kelas_asli]["BENAR"] += 1

                    print("\n" + "="*72)
                    print("  STATUS: [✓] BENAR! (Skor 1)")
                    print(f"  Suara Asli : {kelas_asli}")
                    print(f"  LCD Tampil : {tebakan} (Akurat 100%)")
                    print("  -> Hebat! Tercatat ke Excel sebagai BENAR.")
                    print("="*72)
                else:
                    mode = "SALAH_DETEKSI"
                    skor = 0
                    ket = f"Miskalsifikasi: menebak {tebakan}"
                    confusion[kelas_asli][tebakan] += 1
                    mode_count[kelas_asli]["SALAH_DETEKSI"] += 1
                    retrain_list.append((kelas_asli, fpath, "SALAH DETEKSI", f"Malah menebak {tebakan}"))

                    print("\n" + "!"*72)
                    print("  STATUS: [✗] SALAH DETEKSI (FATAL)! (Skor 0)")
                    print(f"  Suara Asli : {kelas_asli}")
                    print(f"  LCD Tampil : {tebakan}  <--- SALAH KELAS!")
                    print("  -> Tercatat ke Excel dan ditandai untuk evaluasi.")
                    print("!"*72)

            results_log.append({
                "ronde": idx,
                "waktu": waktu_uji,
                "file": fname,
                "kelas_asli": kelas_asli,
                "tebakan": tebakan,
                "mode": mode,
                "skor": skor,
                "keterangan": ket
            })

            # Tampilkan statistik sementara setelah tiap input
            total_tested = len(results_log)
            total_c = sum(1 for r in results_log if r.get("mode") in ("BENAR", "LAMBAT"))
            acc_c = (total_c / total_tested * 100) if total_tested > 0 else 0
            print(f"  Progress: Ronde {idx}/{len(pool)} | Akurasi Sementara: {total_c}/{total_tested} ({acc_c:.1f}%)")

            if idx < len(pool):
                print("-" * 72)
                lanjut_ronde = input("  [Tekan ENTER untuk lanjut ke ronde berikutnya, atau ketik 'q' untuk selesai]: ").strip().lower()
                if lanjut_ronde == 'q':
                    print("\n  [!] Pengujian diakhiri oleh pengguna.")
                    break

    except KeyboardInterrupt:
        print("\n\n" + "="*72)
        print("  [!] Pengujian dihentikan sementara (Ctrl+C).")
        print(f"  Ronde yang telah berhasil diselesaikan: {len(results_log)} suara.")
        print("  Seluruh data yang terkumpul TETAP AKAN DIEKSPOR ke Excel & Grafik!")
        print("="*72)

    # ─────────────────────────────────────────────
    #  LAPORAN AKHIR & EKSPOR EXCEL + GRAFIK
    # ─────────────────────────────────────────────
    os.system('cls' if os.name == 'nt' else 'clear')
    elapsed_total = time.time() - start_time
    total_valid   = sum(sum(c.values()) for c in confusion.values())
    total_benar   = sum(confusion[k][k] for k in KELAS)
    acc_total     = (total_benar / total_valid * 100) if total_valid else 0
    file_missing  = sum(1 for r in results_log if r.get("mode") == "FILE_MISSING")
    skipped       = sum(1 for r in results_log if r.get("mode") == "SKIP")
    total_diam    = sum(mode_count[k]["TIDAK_TERDETEKSI"] for k in KELAS)
    total_salah_k = sum(mode_count[k]["SALAH_DETEKSI"]    for k in KELAS)
    total_lambat  = sum(mode_count[k]["LAMBAT"]           for k in KELAS)

    W = 72
    print()
    print("=" * W)
    print("  LAPORAN STATISTIK LENGKAP PENGUJIAN HARDWARE SIRENMASTER".center(W))
    print("=" * W)
    print(f"  Tanggal          : {time.strftime('%d %B %Y, %H:%M:%S')}")
    print(f"  Durasi Pengujian : {int(elapsed_total//60)} menit {int(elapsed_total%60)} detik")
    print("-" * W)
    print(f"  Total Sampel     : {len(results_log)} suara diuji")
    print(f"  [✓] BENAR        : {total_benar} suara ({acc_total:.1f}%)")
    print(f"  [✗] SALAH KELAS  : {total_salah_k} suara (Fatal misklasifikasi)")
    print(f"  [-] TIDAK RESPON : {total_diam} suara (Missed / System Aman)")
    print(f"  [~] LAMBAT       : {total_lambat} suara (Benar tapi >10 detik)")
    print()
    bar_len = 35
    filled  = int(bar_len * acc_total / 100)
    bar     = "#" * filled + "-" * (bar_len - filled)
    print(f"  AKURASI TOTAL    : [{bar}] {acc_total:.2f}%")
    print("-" * W)

    # Tabel Performa Per Kelas di Terminal
    print("  PERFORMA DETEKSI PER KELAS:")
    print(f"  {'KELAS':<14} | {'BENAR':>6} | {'SALAH':>6} | {'DIAM':>6} | {'LAMBAT':>6} | {'TOTAL':>6} | {'AKURASI':>8}")
    print(f"  {'-'*14}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")
    for k in KELAS:
        tot_k    = sum(confusion[k].values())
        benar_k  = confusion[k][k]
        acc_k    = (benar_k / tot_k * 100) if tot_k else 0
        sk       = mode_count[k]["SALAH_DETEKSI"]
        diam_k   = mode_count[k]["TIDAK_TERDETEKSI"]
        lambat_k = mode_count[k]["LAMBAT"]
        status_k = " (✓ LULUS)" if acc_k >= 90 else " (⚠ PERLU RETRAIN)"
        print(f"  {k:<14} | {benar_k:>6} | {sk:>6} | {diam_k:>6} | {lambat_k:>6} | {tot_k:>6} | {acc_k:>7.1f}%{status_k}")

    # Simpan laporan text & butuh_retrain
    log_path = os.path.join(ROOT, "hasil_benchmark_100.txt")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("LAPORAN BENCHMARK 100 SIRINE -- SIRENMASTER\n")
            f.write("=" * W + "\n")
            f.write(f"Tanggal         : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Durasi          : {int(elapsed_total//60)}m {int(elapsed_total%60)}s\n")
            f.write(f"Akurasi Total   : {total_benar}/{total_valid} = {acc_total:.2f}%\n\n")
            f.write(f"Benar           : {total_benar}\n")
            f.write(f"Salah Kelas     : {total_salah_k}\n")
            f.write(f"Diam            : {total_diam}\n")
            f.write(f"Lambat          : {total_lambat}\n\n")
            f.write("PERFORMA PER KELAS:\n")
            for k in KELAS:
                tot_k = sum(confusion[k].values())
                bk = confusion[k][k]
                acc_k = (bk / tot_k * 100) if tot_k else 0
                f.write(f"  {k:<14} : {bk}/{tot_k} ({acc_k:.1f}%)\n")
    except Exception as e:
        print(f"  [!] Gagal simpan log txt: {e}")

    # ── KONFIRMASI SIMPAN KE FOLDER PENGUJIAN (Pengujian_1, Pengujian_2, ...) ──
    next_num, next_folder = get_next_pengujian_name()

    print("\n" + "=" * W)
    print("  SIMPAN HASIL PENGUJIAN KE ARSIP SKRIPSI?".center(W))
    print("=" * W)
    print(f"  Target Folder : [ {next_folder} ]")
    print(f"  Pilihan Tindakan:")
    print(f"   [1] YA, Simpan ke folder '{next_folder}' (Lengkap: Excel Dashboard + 4 Grafik + Analisis)")
    print(f"   [2] YA, Simpan dengan nama folder kustom")
    print(f"   [0] TIDAK, Jangan simpan sesi ini ke arsip")
    print("=" * W)

    try:
        pilih_save = input(f"  Pilihan Anda [1=Simpan {next_folder} / 2=Kustom / 0=Jangan Simpan]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n\n  [!] Input tidak tersedia. Data tidak disimpan — kembali ke menu utama.")
        return

    if pilih_save == "0":
        print("\n  [-] Pengujian tidak disimpan ke arsip.")
        try:
            input("\n[Tekan ENTER untuk kembali ke menu utama...]")
        except (EOFError, KeyboardInterrupt):
            pass
        return
    elif pilih_save == "2":
        try:
            kustom_nama = input("  Masukkan nama folder (contoh: Pengujian_1, Uji_Pagi, dll): ").strip()
        except (EOFError, KeyboardInterrupt):
            kustom_nama = ""
        target_folder = kustom_nama if kustom_nama else next_folder
    else:
        target_folder = next_folder

    print("\n" + "=" * W)
    print(f"  MENYIMPAN DATA & MEMBUAT DASHBOARD KE FOLDER: {target_folder}...")
    print("=" * W)

    export_res = None
    try:
        export_res = export_benchmark_to_excel_and_charts(
            results_log, confusion, mode_count, elapsed_total, folder_name=target_folder
        )
        print(f"  [✓] Folder Pengujian Berhasil Dibuat: {export_res['session_dir']}")
        print(f"  [✓] File Excel Dashboard Data Analysis: {os.path.basename(export_res['excel_path'])}")
        print(f"  [✓] Ringkasan Naratif: {os.path.basename(export_res['txt_path'])}")
        print("  [✓] 4 Grafik Skripsi 300 DPI Berhasil Dibuat di subfolder 'grafik/':")
        for g_name, g_path in export_res['chart_paths'].items():
            print(f"      • {os.path.basename(g_path)}")
    except Exception as e:
        print(f"  [!] Gagal ekspor Excel/Grafik: {e}")

    # Simpan file yang butuh retrain jika ada
    if retrain_list:
        retrain_path = os.path.join(export_res['session_dir'] if export_res else ROOT, "butuh_retrain.txt")
        try:
            with open(retrain_path, "w", encoding="utf-8") as f:
                f.write("DAFTAR FILE BERMASALAH (BUTUH LATIH ULANG)\n")
                f.write("=" * 70 + "\n")
                f.write(f"Tanggal : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total   : {len(retrain_list)} file\n\n")
                for kls in KELAS:
                    grup = [(fp, md, al) for (ka, fp, md, al) in retrain_list if ka == kls]
                    if not grup:
                        continue
                    f.write(f"\n[{kls}] — {len(grup)} file:\n")
                    f.write("-" * 60 + "\n")
                    for fpath, mode, alasan in grup:
                        f.write(f"  File    : {os.path.basename(fpath)}\n")
                        f.write(f"  Masalah : {mode} ({alasan})\n")
                        f.write(f"  Path    : {fpath}\n\n")
            print(f"\n  [*] Daftar {len(retrain_list)} file bermasalah tersimpan di: {os.path.basename(retrain_path)}")
        except Exception:
            pass

    print("=" * W)
    if export_res:
        print("  AKSI CEPAT:")
        print(f"   [1] Buka File Excel Dashboard Sekarang (Microsoft Excel)")
        print(f"   [2] Buka Folder '{target_folder}' (Windows Explorer)")
        print("   [ENTER] Selesai & Kembali ke Menu Utama")
        try:
            aksi = input("  Pilih aksi [1/2/ENTER]: ").strip()
        except (EOFError, KeyboardInterrupt):
            aksi = ""
        if aksi == "1":
            try:
                os.startfile(export_res['excel_path'])
            except Exception as e:
                print(f"  [!] Tidak dapat membuka Excel otomatis: {e}")
        elif aksi == "2":
            try:
                os.startfile(export_res['session_dir'])
            except Exception as e:
                print(f"  [!] Tidak dapat membuka folder: {e}")
    try:
        input("\n[Tekan ENTER untuk kembali ke menu utama...]")  
    except (EOFError, KeyboardInterrupt):
        pass

def uji_suara_normal():
    """Menu [6]: Putar suara Normal/Bising secara random."""
    normal_dir = os.path.join(ROOT, "NORMAL")
    if not os.path.exists(normal_dir):
        print("\n[!] Folder NORMAL tidak ditemukan!")
        return
        
    wav_files = [f for f in os.listdir(normal_dir) if f.endswith(".wav")]
    if not wav_files:
        print("\n[!] Tidak ada file .wav di folder NORMAL!")
        return
        
    print("\n" + "="*65)
    print(" UJIAN SUARA PENGECOH (NORMAL / BISING)")
    print("="*65)
    print(f" Total file tersedia di folder NORMAL: {len(wav_files)} suara.")
    
    try:
        jml = input(" Masukkan jumlah suara yang ingin diputar (contoh: 10, 50, 100): ").strip()
        jml = int(jml)
        if jml <= 0: return
    except ValueError:
        print(" [!] Harus berupa angka!")
        return
        
    print(f"\n Memutar {jml} suara pengecoh secara acak.")
    print(" PASTIKAN ALAT ANDA TETAP AMAN (TIDAK MENGUNCI)!")
    print()
    
    # Memilih sampel secara acak dengan penggantian (allow duplicates) jika jml > len(wav_files)
    samples = random.choices(wav_files, k=jml)
    
    for i, file in enumerate(samples, 1):
        fpath = os.path.join(normal_dir, file)
        print(f"\n [{i}/{jml}] Memutar: {file}")
        for putaran in range(1, 4):
            print(f"        -> Putaran {putaran}/3...", end="\r", flush=True)
            winsound.PlaySound(fpath, winsound.SND_FILENAME)
            time.sleep(0.1)
        print(f"        -> Selesai diputar 3x (Total ~12 Detik).")
        time.sleep(1.0)
        
    print("\n [+] UJIAN SELESAI!")
    print(" Jika layar alat Anda tidak pernah mengunci ke Sirine (Tetap SAFE),")
    print(" berarti AI Anda sudah 100% KEBAL / ANTI BOCOR!")
    print("="*65)

def main():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=================================================================")
        print(" SIRENMASTER AUDIO TESTER - PENGUJIAN & UJIAN SKRIPSI")
        print("=================================================================")
        print("  Pilih menu pengujian berikut:")
        print()
        print("   [1]  PUTAR 1 SUARA TUNGGAL (Pilih Ambulance/Damkar/Polisi)")
        print("   [2]  UJIAN 10 SUARA SIRINE (10 tes berurutan)")
        print("   [3]  UJIAN RANDOM & PENCATATAN SKOR")
        print("   [4]  BENCHMARK 100 SIRINE + STATISTIK LENGKAP")
        print("   [5]  REKAM SUARA MANUAL (Buka Aplikasi Perekam)")
        print("   [6]  UJI SUARA PENGECOH (Random Suara Bising / Normal)")
        print()
        print("   [0]  KELUAR")
        print("=================================================================")

        pilihan = input("Masukkan nomor pilihan Anda (0-6): ").strip()

        if pilihan == "0":
            print("\n[+] Terima kasih! Semoga sukses ujian skripsinya!")
            break

        elif pilihan == "1":
            print("\n--- PILIH SUARA TUNGGAL ---")
            print(" [a] AMBULANCE")
            print(" [b] DAMKAR")
            print(" [c] POLISI")
            sub = input("Pilih a, b, atau c: ").strip().lower()
            key = "1" + sub
            if key in SINGLE_SAMPLES:
                cls_name, filepath = SINGLE_SAMPLES[key]
                play_sound(cls_name, filepath, loops=4)
            else:
                print("[!] Pilihan tidak valid.")
            input("\n[Tekan ENTER untuk kembali ke menu utama...]")

        elif pilihan == "2":
            print("\n" + "#"*65)
            print(" MEMULAI UJIAN 10 SUARA SIRINE RANDOM (DIPUTAR 4X PER SUARA)")
            print("#"*65)
            # Dibuat BARU setiap kali menu ini dipilih! Selalu berbeda!
            exam_samples = build_exam_10()
            print(f" [INFO] {len(exam_samples)} suara dipilih secara acak dari dataset.")
            for i, (cname, fpath) in enumerate(exam_samples, 1):
                print(f"\n----> [TES #{i} dari {len(exam_samples)}] : {cname} <----")
                play_sound(cname, fpath, loops=4)
                if i < len(exam_samples):
                    input(f"\n[Catat hasil LCD! Tekan ENTER untuk lanjut ke Tes #{i+1}...] ")
            input("\n[Seluruh tes ujian selesai! Tekan ENTER ke menu utama...]")


        elif pilihan == "3":
            print("\n" + "#"*65)
            print(" UJIAN RANDOM 50 SIRINE - HANYA Y / N")
            print("#"*65)
            n_rounds = 50

            score_correct = 0
            score_wrong   = 0
            history       = []
            pool          = list(RANDOM_POOL_15)

            # Per-kelas tracking
            kelas_stat = {"AMBULANCE": [0,0], "FIRETRUCK": [0,0], "POLISI": [0,0]}

            for r in range(1, n_rounds + 1):
                cname, fpath = random.choice(pool)
                print(f"\n{'='*55}")
                print(f" RONDE {r:02d}/{n_rounds}  |  Kelas: [{cname}]")
                print(f"{'='*55}")
                print(" Memutar 4x... Dekatkan alat ke speaker!")
                for lp in range(1, 5):
                    print(f"    -> Putaran {lp}/4...", end="\r")
                    winsound.PlaySound(fpath, winsound.SND_FILENAME)
                    time.sleep(0.1)
                print(f"\n [+] Selesai diputar!")
                print(f"\n  Apakah alat mendeteksi [{cname}] dengan benar?")
                print("  [Y] = BENAR    [N] = SALAH")
                while True:
                    ans = input("  Ketik Y atau N: ").strip().lower()
                    if ans in ['y', 'n']:
                        break
                    print("  [!] Hanya ketik Y atau N!")

                if ans == 'y':
                    score_correct += 1
                    status_str = "BENAR ✓"
                    kelas_stat.setdefault(cname, [0,0])[0] += 1
                else:
                    score_wrong += 1
                    status_str = "SALAH ✗"
                kelas_stat.setdefault(cname, [0,0])[1] += 1

                history.append((r, cname, status_str))
                total_so_far = score_correct + score_wrong
                akurasi_now  = score_correct / total_so_far * 100 if total_so_far > 0 else 0
                print(f"\n  => {status_str}  |  Skor sementara: {score_correct}/{total_so_far} ({akurasi_now:.1f}%)")
                time.sleep(0.3)

            # ── LAPORAN AKHIR ──
            total_valid = score_correct + score_wrong
            akurasi     = score_correct / total_valid * 100 if total_valid > 0 else 0
            bar_len     = int(akurasi / 2.5)
            bar         = "#" * bar_len + "-" * (40 - bar_len)

            print("\n\n" + "="*65)
            print(" LAPORAN AKHIR UJIAN 50 SIRINE")
            print("="*65)
            print(f"  Total Ronde  : {total_valid}")
            print(f"  BENAR        : {score_correct}")
            print(f"  SALAH        : {score_wrong}")
            print(f"  AKURASI      : [{bar}] {akurasi:.1f}%")
            print()
            print("  PERFORMA PER KELAS:")
            print(f"  {'KELAS':<14} | BENAR | SALAH | TOTAL | AKURASI")
            print(f"  {'-'*14}-+-------+-------+-------+--------")
            for k, (b, tot) in kelas_stat.items():
                if tot == 0: continue
                s = tot - b
                ak = b / tot * 100
                print(f"  {k:<14} | {b:>5} | {s:>5} | {tot:>5} | {ak:>6.1f}%")
            print()
            print("  DETAIL TIAP RONDE:")
            for r_num, c_str, s_str in history:
                print(f"   Ronde {r_num:02d}: {c_str:<14} -> {s_str}")
            print("="*65)
            input("\n[Tekan ENTER untuk kembali ke menu utama...] ")


        elif pilihan == "4":
            menu_ujian_100()
            
        elif pilihan == "5":
            print("\n[+] Membuka Aplikasi Perekam Manual...")
            rekam_path = os.path.join(ROOT, "05_Testing_dan_Live", "rekam_manual.py")
            if os.path.exists(rekam_path):
                subprocess.run(["python", rekam_path])
            else:
                print(f"[!] File {rekam_path} tidak ditemukan!")
            input("\n[Tekan ENTER untuk kembali ke menu utama...]")
            
        elif pilihan == "6":
            uji_suara_normal()
            input("\n[Tekan ENTER untuk kembali ke menu utama...]")

        else:
            print("[!] Pilihan tidak valid, silakan coba lagi.")
            time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[+] Program dihentikan. Sampai jumpa!")
