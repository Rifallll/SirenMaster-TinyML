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


def build_pool_100():
    """Bangun 100 sampel uji seimbang per kelas (33 AMB + 33 DAMKAR + 34 POLISI).
    Urutan DIACAK agar pengujian real seperti kondisi nyata.
    Seed = waktu sekarang (setiap run berbeda)."""

    def get_seg_files(folder_name, cls_label, target_n):
        """Ambil file .wav murni (bukan aug/synth/noise), ulangi jika kurang dari target."""
        folder_path = os.path.join(ROOT, folder_name)
        if not os.path.exists(folder_path):
            return []
        prefix_map = {
            "AMBULANCE": "ambulance_",
            "FIRETRUCK": "fire_",
            "POLICE":    "police_"
        }
        pref = prefix_map.get(folder_name, "")
        files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith("_seg01.wav")
            and f.lower().startswith(pref)
            and not f.startswith("aug_")
            and not f.startswith("SYNTH_")
            and "noise" not in f
            and "shift" not in f
            and "loud"  not in f
        ])
        if not files:
            files = sorted([
                f for f in os.listdir(folder_path)
                if f.endswith(".wav")
                and f.lower().startswith(pref)
                and not f.startswith("aug_")
                and not f.startswith("SYNTH_")
                and "noise" not in f
                and "shift" not in f
            ])
        if not files:
            return []

        base = [(cls_label, os.path.join(folder_path, f)) for f in files]
        out  = []
        while len(out) < target_n:
            out.extend(base)
        return out[:target_n]

    pool  = get_seg_files("AMBULANCE", "AMBULANCE", 33)
    pool += get_seg_files("FIRETRUCK", "DAMKAR",    33)
    pool += get_seg_files("POLICE",    "POLISI",    34)

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
    """Menu [4]: Benchmark 100 Sirine dengan Statistik Lengkap per Kelas."""
    KELAS = ["AMBULANCE", "DAMKAR", "POLISI"]
    try:
        pool = build_pool_100()
    except KeyboardInterrupt:
        print("\n\n[!] Dibatalkan. Kembali ke menu utama.")
        return
    pool_backup = list(pool)  # simpan jika ingin resume

    if not pool:
        print("\n[!] POOL 100 KOSONG.")
        print("    Pastikan folder AMBULANCE, FIRETRUCK, POLICE berisi file .wav.")
        input("\n[Tekan ENTER untuk kembali ke menu utama...]")
        return

    os.system('cls')
    print("="*70)
    print("  BENCHMARK 100 SIRINE  --  DATA RESMI SKRIPSI")
    print("="*70)
    print(f"  Total sampel : {len(pool)} suara")
    print("  Distribusi   : 33 Ambulance | 33 Damkar | 34 Polisi")
    print("  Urutan       : ACAK setiap run (berbeda tiap kali dijalankan)")
    print("-"*70)
    print("  CARA KERJA:")
    print("  1. Sistem memutar suara 2x -> Dekatkan ESP32 ke speaker.")
    print("  2. Ketik kelas yang tampil di LCD alat Anda.")
    print("  3. Di akhir: Confusion Matrix + Akurasi Per Kelas + File Laporan.")
    print("="*70)
    input("\n[Tekan ENTER untuk mulai Ronde 1 dari 100...]")


    # ── Struktur data: confusion + mode kegagalan ──────────────────
    # Tuple results_log: (ronde, kelas_asli, tebakan_lcd, mode)
    # mode: BENAR | SALAH_DETEKSI | TIDAK_TERDETEKSI | LAMBAT | SKIP | FILE_MISSING
    confusion   = {k: collections.Counter() for k in KELAS + ["TDK_DETEKSI"]}
    mode_count  = {k: collections.Counter() for k in KELAS}
    results_log = []
    retrain_list = []  # [(kelas_asli, fpath, mode, alasan)] — file yg perlu dilatih ulang
    start_time  = time.time()

    try:
        for idx, (kelas_asli, fpath) in enumerate(pool, 1):
            os.system('cls')
            print("="*70)
            print(f"  RONDE {idx:>3} / {len(pool)}")
            print(f"  Kelas Asli : {kelas_asli}")
            print(f"  File       : {os.path.basename(fpath)}")
            print("="*70)
            print("  Memutar 2x... Dekatkan alat ke speaker!\n")

            file_ok = os.path.exists(fpath)
            if not file_ok:
                print(f"    [!] File tidak ditemukan — Ronde {idx} dilewati otomatis.")
                results_log.append((idx, kelas_asli, "-", "FILE_MISSING"))  # FIX: mode di posisi ke-4
                time.sleep(0.4)
                if idx < len(pool):
                    input(f"  [ENTER untuk Ronde {idx+1}...]")
                continue

            # ── Putar audio 3x ──
            for lp in range(1, 4):
                print(f"    -> Putaran {lp}/3...")
                winsound.PlaySound(fpath, winsound.SND_FILENAME)
                time.sleep(0.1)

            # ── Tampilkan prompt dengan konteks kelas yang diuji ──
            cls_icon = {"AMBULANCE": "[A]", "DAMKAR": "[D]", "POLISI": "[P]"}
            print("\n" + "="*70)
            print(f"  Yang diputar   : {kelas_asli}")
            print(f"  Yang HARUS muncul di LCD: {kelas_asli}  {cls_icon.get(kelas_asli,'')}")
            print("-"*70)
            print("  LCD menampilkan APA? Ketik 1 huruf:")
            print()
            print("  [a]  AMBULANCE muncul")
            print("  [d]  DAMKAR    muncul")
            print("  [p]  POLISI    muncul")
            print()
            print("  [0]  DIAM -- LCD tidak bereaksi (masih SYSTEM AMAN)")
            print("  [l]  LAMBAT -- Benar tapi lama terkunci (> 10 detik)")
            print("  [x]  SKIP ronde ini")
            print("="*70)

            while True:
                ans = input("  Ketik (a/d/p/0/l/x): ").strip().lower()
                if ans in ('a', 'd', 'p', '0', 'l', 'x'):
                    break
                print("  [!] Tidak valid. Ketik a, d, p, 0, l, atau x.")

            map_lcd = {'a': 'AMBULANCE', 'd': 'DAMKAR', 'p': 'POLISI'}

            if ans == 'x':
                print(f"  => Ronde {idx} dilewati.")
                results_log.append((idx, kelas_asli, "-", "SKIP"))
                time.sleep(0.2)
                if idx < len(pool):
                    input(f"  [ENTER untuk Ronde {idx+1}...]")
                continue

            elif ans == '0':
                tebakan = "TDK_DETEKSI"
                mode    = "TIDAK_TERDETEKSI"
                confusion[kelas_asli]["TDK_DETEKSI"] += 1
                mode_count[kelas_asli]["TIDAK_TERDETEKSI"] += 1
                retrain_list.append((kelas_asli, fpath, "TIDAK TERDETEKSI", "Layar diam / Sistem Aman"))
                print()
                print("  " + "~"*66)
                print(f"  [-] TIDAK TERDETEKSI! -- {kelas_asli} terlewat")
                print(f"       Seharusnya : {kelas_asli}")
                print(f"       LCD tampil : SYSTEM AMAN (tidak bereaksi)")
                print(f"  [*] FILE INI DITANDAI untuk dilatih ulang ({len(retrain_list)} file sejauh ini)")
                print("  " + "~"*66)

            elif ans == 'l':
                tebakan = kelas_asli
                mode    = "LAMBAT"
                confusion[kelas_asli][kelas_asli] += 1
                mode_count[kelas_asli]["LAMBAT"] += 1
                print()
                print(f"  [ok-lambat] {kelas_asli} terdeteksi tapi LAMBAT (dihitung benar)")

            else:
                # ans adalah a / d / p
                tebakan = map_lcd[ans]
                if tebakan == kelas_asli:
                    mode = "BENAR"
                    confusion[kelas_asli][kelas_asli] += 1
                    mode_count[kelas_asli]["BENAR"] += 1
                    print()
                    print("  " + "-"*66)
                    print(f"  [v] BENAR -- LCD tampil {tebakan} (tepat!)")
                    print("  " + "-"*66)
                else:
                    mode = "SALAH_DETEKSI"
                    confusion[kelas_asli][tebakan] += 1
                    mode_count[kelas_asli]["SALAH_DETEKSI"] += 1
                    retrain_list.append((kelas_asli, fpath, "SALAH DETEKSI", f"Malah menebak {tebakan}"))
                    print()
                    print("  " + "!"*66)
                    print(f"  [X] SALAH DETEKSI (FATAL)!")
                    print(f"       Seharusnya : {kelas_asli}")
                    print(f"       LCD tampil : {tebakan}  <-- SALAH!")
                    print(f"  [*] FILE INI DITANDAI untuk dilatih ulang ({len(retrain_list)} file sejauh ini)")
                    print("  " + "!"*66)

            # Progress kilat setiap 10 ronde
            # BUG FIX: hitung done_so_far hanya dari kelas valid (bukan TDK_DETEKSI)
            done_so_far  = sum(mode_count[k]["BENAR"] + mode_count[k]["SALAH_DETEKSI"]
                               + mode_count[k]["TIDAK_TERDETEKSI"] + mode_count[k]["LAMBAT"]
                               for k in KELAS)
            benar_so_far = sum(confusion[k][k] for k in KELAS)
            pct          = (benar_so_far / done_so_far * 100) if done_so_far else 0
            if idx % 10 == 0:
                elapsed = time.time() - start_time
                eta     = (elapsed / idx) * (len(pool) - idx) if idx < len(pool) else 0
                print(f"\n  -- [{idx}/{len(pool)}] Skor: {benar_so_far}/{done_so_far} = {pct:.1f}% | ETA ~{eta/60:.1f} mnt --")

            results_log.append((idx, kelas_asli, tebakan, mode))
            time.sleep(0.2)
            if idx < len(pool):
                input(f"  [ENTER untuk Ronde {idx+1}...]")

    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("  [!] Pengujian dihentikan manual (Ctrl+C).")
        print(f"  Ronde yang sudah selesai: {len(results_log)}")
        lanjut = input("  Tetap tampilkan laporan parsial? (y/n): ").strip().lower()
        if lanjut != 'y':
            print("  Laporan dibatalkan. Kembali ke menu utama.")
            return

    # ─────────────────────────────────────────────
    #  LAPORAN AKHIR
    # ─────────────────────────────────────────────
    os.system('cls')
    elapsed_total = time.time() - start_time
    total_valid   = sum(sum(c.values()) for c in confusion.values())
    total_benar   = sum(confusion[k][k] for k in KELAS)
    acc_total     = (total_benar / total_valid * 100) if total_valid else 0
    file_missing  = sum(1 for _,_,_,m in results_log if m == "FILE_MISSING")
    skipped       = sum(1 for _,_,_,m in results_log if m == "SKIP")
    total_diam    = sum(mode_count[k]["TIDAK_TERDETEKSI"] for k in KELAS)
    total_salah_k = sum(mode_count[k]["SALAH_DETEKSI"]    for k in KELAS)
    total_lambat  = sum(mode_count[k]["LAMBAT"]           for k in KELAS)

    W = 70
    print()
    print("=" * W)
    print("  LAPORAN STATISTIK LENGKAP".center(W))
    print("  BENCHMARK 100 SIRINE -- SIRENMASTER".center(W))
    print("=" * W)
    print(f"  Tanggal          : {time.strftime('%d %B %Y, %H:%M:%S')}")
    print(f"  Durasi Pengujian : {int(elapsed_total//60)} menit {int(elapsed_total%60)} detik")
    print("-" * W)
    print(f"  Total Ronde      : 100")
    print(f"  Ronde Diuji      : {total_valid}")
    print(f"  Skip / Missing   : {skipped} skip + {file_missing} file hilang")
    print()
    print(f"  [v] BENAR            : {total_benar}")
    print(f"  [X] SALAH DETEKSI    : {total_salah_k}  (Model menebak kelas lain - FATAL)")
    print(f"  [-] TIDAK TERDETEKSI : {total_diam}  (Layar diam / Missed)")
    print(f"  [~] LAMBAT           : {total_lambat}  (Benar tapi butuh waktu >10 detik)")
    print()
    bar_len = 40
    filled  = int(bar_len * acc_total / 100)
    bar     = "#" * filled + "-" * (bar_len - filled)
    print(f"  AKURASI TOTAL : [{bar}] {acc_total:.2f}%")

    # ── Akurasi + Mode Kegagalan Per Kelas ──
    print()
    print("-" * W)
    print("  PERFORMA PER KELAS:")
    print()
    col_kelas = "KELAS"
    print(f"  {col_kelas:<14} | {'BENAR':>6} | {'SALAH_D':>7} | {'TDK_DET':>7} | {'LAMBAT':>6} | {'TOTAL':>6} | {'AKURASI':>8}")
    print(f"  {'-'*14}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")
    for k in KELAS:
        tot_k    = sum(confusion[k].values())
        benar_k  = confusion[k][k]
        acc_k    = (benar_k / tot_k * 100) if tot_k else 0
        sk       = mode_count[k]["SALAH_DETEKSI"]
        diam_k   = mode_count[k]["TIDAK_TERDETEKSI"]
        lambat_k = mode_count[k]["LAMBAT"]
        flag     = " <-- PERLU PERBAIKAN" if acc_k < 80 else ("⚠" if acc_k < 90 else "")
        print(f"  {k:<14} | {benar_k:>6} | {sk:>7} | {diam_k:>7} | {lambat_k:>6} | {tot_k:>6} | {acc_k:>7.1f}% {flag}")

    # ── Confusion Matrix ──
    print()
    print("-" * W)
    print("  CONFUSION MATRIX  (Baris=Kelas Asli | Kolom=Tebakan LCD)")
    print("  Nilai diagonal (*) = benar. Di luar diagonal = salah deteksi.")
    print()
    col0   = "Asli / LCD"
    header = f"  {col0:<14} | {'AMB':>6} | {'DAM':>6} | {'POL':>6} | {'TDK_DET':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k in KELAS:
        r_amb  = confusion[k]["AMBULANCE"]
        r_dam  = confusion[k]["DAMKAR"]
        r_pol  = confusion[k]["POLISI"]
        r_diam = confusion[k]["TDK_DETEKSI"]
        vals   = [
            f"{r_amb}*" if k == "AMBULANCE" else str(r_amb),
            f"{r_dam}*" if k == "DAMKAR"    else str(r_dam),
            f"{r_pol}*" if k == "POLISI"    else str(r_pol),
            str(r_diam),
        ]
        print(f"  {k:<14} | {vals[0]:>6} | {vals[1]:>6} | {vals[2]:>6} | {vals[3]:>6}")

    # ── Analisis Kelemahan Otomatis ──
    print()
    print("-" * W)
    print("  ANALISIS KELEMAHAN OTOMATIS:")
    any_issue = False
    for k in KELAS:
        tot_k  = sum(confusion[k].values())
        acc_k  = (confusion[k][k] / tot_k * 100) if tot_k else 0
        diam_k = mode_count[k]["TIDAK_TERDETEKSI"]
        sk_k   = mode_count[k]["SALAH_DETEKSI"]

        issues = []
        if diam_k > 0:
            issues.append(f"TDK DETEKSI {diam_k}x")
        if sk_k > 0:
            top = sorted([(c, n) for c, n in confusion[k].items()
                          if c != k and c != "TDK_DETEKSI" and n > 0], key=lambda x: -x[1])
            if top:
                issues.append(f"SALAH DETEKSI {sk_k}x (sering dikira '{top[0][0]}')")
        if mode_count[k]["LAMBAT"] > 0:
            issues.append(f"LAMBAT {mode_count[k]['LAMBAT']}x")

        if issues:
            any_issue = True
            print(f"  {k:<12} [{acc_k:.1f}%]: {' | '.join(issues)}")

    if not any_issue:
        print("  Tidak ada masalah signifikan. Semua kelas performa baik!")

    # ── Simpan laporan TXT ──
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
            f.write(f"  {'KELAS':<14} | {'BENAR':>6} | {'SALAH_D':>7} | {'TDK_DET':>7} | {'LAMBAT':>6} | {'AKURASI':>8}\n")
            for k in KELAS:
                tot_k  = sum(confusion[k].values())
                bk     = confusion[k][k]
                acc_k  = (bk / tot_k * 100) if tot_k else 0
                sk     = mode_count[k]["SALAH_DETEKSI"]
                dk     = mode_count[k]["TIDAK_TERDETEKSI"]
                lk     = mode_count[k]["LAMBAT"]
                f.write(f"  {k:<14} | {bk:>6} | {sk:>7} | {dk:>7} | {lk:>6} | {acc_k:>7.1f}%\n")
            f.write("\nCONFUSION MATRIX:\n")
            f.write(f"  {'Asli/LCD':<14} | {'AMB':>6} | {'DAM':>6} | {'POL':>6} | {'TDK_DET':>7}\n")
            for k in KELAS:
                f.write(f"  {k:<14} | {confusion[k]['AMBULANCE']:>6} | {confusion[k]['DAMKAR']:>6}"
                        f" | {confusion[k]['POLISI']:>6} | {confusion[k]['TDK_DETEKSI']:>7}\n")
            f.write("\nDETAIL PER RONDE:\n")
            f.write(f"  {'No':>4} | {'Kelas Asli':<12} | {'LCD Tampil':<12} | MODE\n")
            f.write("  " + "-" * 52 + "\n")
            for r_num, k_asli, tebakan, mode_r in results_log:
                f.write(f"  {r_num:>4} | {k_asli:<12} | {tebakan:<12} | {mode_r}\n")
        print(f"\n  Laporan disimpan: {log_path}")
    except Exception as e:
        print(f"\n  [!] Gagal simpan laporan: {e}")

    # ── Tampilkan & Simpan file yang butuh retrain ──
    print()
    print("=" * W)
    if retrain_list:
        retrain_list_sorted = sorted(retrain_list, key=lambda x: (x[0], x[2]))

        # ── Tampilkan di terminal ──
        print(f"  FILE BERMASALAH (BUTUH LATIH ULANG) — Total: {len(retrain_list)} file")
        print("=" * W)

        # Kelompokkan per kelas
        for kls in ["AMBULANCE", "DAMKAR", "POLISI"]:
            grup = [(fp, md, al) for (ka, fp, md, al) in retrain_list_sorted if ka == kls]
            if not grup:
                continue
            icon = {"AMBULANCE": "[A]", "DAMKAR": "[D]", "POLISI": "[P]"}.get(kls, "")
            print(f"\n  {icon} {kls}  ({len(grup)} file):")
            print("  " + "-" * 60)
            for fpath, mode, alasan in grup:
                fname = os.path.basename(fpath)
                tag   = "TIDAK TERDETEKSI" if mode == "TIDAK TERDETEKSI" else f"SALAH DETEKSI ({alasan})"
                print(f"    • {fname}")
                print(f"      Masalah : {tag}")

        print()
        print("  Saran tindakan:")
        print("  1. Upload file-file di atas ke Edge Impulse (kelas masing-masing)")
        print("  2. Retrain model → Deploy ulang ke ESP32")
        print("  3. Jalankan benchmark lagi untuk validasi peningkatan")
        print()

        # ── Simpan ke file ──
        retrain_path = os.path.join(ROOT, "butuh_retrain.txt")
        try:
            with open(retrain_path, "w", encoding="utf-8") as f:
                f.write("DAFTAR FILE BERMASALAH (BUTUH LATIH ULANG)\n")
                f.write("=" * 70 + "\n")
                f.write(f"Tanggal : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total   : {len(retrain_list)} file\n\n")
                for kls in ["AMBULANCE", "DAMKAR", "POLISI"]:
                    grup = [(fp, md, al) for (ka, fp, md, al) in retrain_list_sorted if ka == kls]
                    if not grup:
                        continue
                    f.write(f"\n[{kls}] — {len(grup)} file:\n")
                    f.write("-" * 60 + "\n")
                    for fpath, mode, alasan in grup:
                        f.write(f"  File    : {os.path.basename(fpath)}\n")
                        f.write(f"  Masalah : {mode} ({alasan})\n")
                        f.write(f"  Path    : {fpath}\n\n")
            print(f"  File lengkap disimpan: {retrain_path}")
        except Exception as e:
            print(f"  [!] Gagal simpan: {e}")
    else:
        print("  Tidak ada file bermasalah! Semua kelas terdeteksi dengan benar.")

    print("=" * W)
    input("\n[Benchmark selesai! Tekan ENTER untuk kembali ke menu utama...]")

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
