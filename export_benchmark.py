"""
export_benchmark.py
===================
Modul otomatis untuk mengekspor hasil pengujian manual hardware SirenMaster ke:
1. Folder terorganisir per sesi (Pengujian_1, Pengujian_2, Pengujian_3, dst.)
2. File Excel (.xlsx) dengan Executive Dashboard Data Analysis & Data Mentah.
3. Grafik visualisasi (.png) resolusi tinggi 300 DPI di dalam subfolder 'grafik/':
   - Dashboard Komprehensif (3-in-1 multi-panel)
   - Bar Chart Akurasi per Kelas
   - Confusion Matrix Heatmap
   - Donut Chart Distribusi Status Pengujian
4. File ringkasan teks analitik (Ringkasan_Analisis_Pengujian_X.txt) untuk Bab 4 Skripsi.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os
import re
import time
import shutil
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as OpenpyxlImage
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = r"C:\Users\ASUS\Videos\DATASET\HASIL_PENGUJIAN_SKRIPSI"


def get_next_pengujian_name(base_dir=OUTPUT_DIR):
    """
    Mendeteksi subfolder yang sudah ada (Pengujian_1, Pengujian_2, ...)
    dan mengembalikan nomor berikutnya beserta nama foldernya.
    """
    os.makedirs(base_dir, exist_ok=True)
    existing = os.listdir(base_dir)
    max_num = 0
    for item in existing:
        full = os.path.join(base_dir, item)
        if os.path.isdir(full):
            m = re.match(r'^Pengujian[_\s](\d+)$', item, re.IGNORECASE)
            if m:
                num = int(m.group(1))
                if num > max_num:
                    max_num = num
    next_num = max_num + 1
    return next_num, f"Pengujian_{next_num}"


def generate_charts(confusion, mode_count, overall_accuracy, timestamp_str, target_grafik_dir):
    """
    Menghasilkan 4 grafik visualisasi berkualitas tinggi (.png) ke dalam target_grafik_dir
    """
    os.makedirs(target_grafik_dir, exist_ok=True)
    kelas_list = ["AMBULANCE", "DAMKAR", "POLISI"]
    chart_paths = {}

    acc_per_class = []
    tot_per_class = []
    benar_per_class = []
    salah_per_class = []
    diam_per_class = []

    for k in kelas_list:
        b = confusion.get(k, {}).get(k, 0)
        tot = sum(confusion.get(k, {}).values())
        acc = (b / tot * 100) if tot > 0 else 0
        s = mode_count.get(k, {}).get("SALAH_DETEKSI", 0)
        d = mode_count.get(k, {}).get("TIDAK_TERDETEKSI", 0)

        acc_per_class.append(acc)
        tot_per_class.append(tot)
        benar_per_class.append(b)
        salah_per_class.append(s)
        diam_per_class.append(d)

    # -------------------------------------------------------------
    # 1. GRAFIK AKURASI PER KELAS (BAR CHART)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    categories = kelas_list + ["RATA-RATA TOTAL"]
    scores = acc_per_class + [overall_accuracy]
    colors = ['#1F4E79', '#C00000', '#2E75B6', '#385723']

    bars = ax.bar(categories, scores, color=colors, width=0.52, edgecolor='#222222', linewidth=1.2, zorder=3)
    ax.set_ylim(0, 108)
    ax.set_ylabel('Akurasi Pengujian (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('AKURASI DETEKSI HARDWARE SIRENMASTER PER KELAS', fontsize=13, fontweight='bold', pad=15)
    ax.axhline(90, color='#D9534F', linestyle='--', linewidth=1.5, label='Batas Kelulusan Skripsi (90%)', zorder=2)
    ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(f'{score:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    bar_path = os.path.join(target_grafik_dir, "grafik_akurasi_per_kelas.png")
    fig.savefig(bar_path)
    plt.close(fig)
    chart_paths['bar_akurasi'] = bar_path

    # -------------------------------------------------------------
    # 2. CONFUSION MATRIX HEATMAP
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    matrix_data = []
    cols_display = ["AMBULANCE", "DAMKAR", "POLISI", "TDK_DETEKSI"]
    for k in kelas_list:
        row = [
            confusion.get(k, {}).get("AMBULANCE", 0),
            confusion.get(k, {}).get("DAMKAR", 0),
            confusion.get(k, {}).get("POLISI", 0),
            confusion.get(k, {}).get("TDK_DETEKSI", 0)
        ]
        matrix_data.append(row)
    matrix_np = np.array(matrix_data)

    cax = ax.matshow(matrix_np, cmap='Blues', alpha=0.85)
    fig.colorbar(cax, shrink=0.8)

    ax.set_xticks(range(len(cols_display)))
    ax.set_yticks(range(len(kelas_list)))
    ax.set_xticklabels(["Ambulans", "Damkar", "Polisi", "Diam (0)"], fontsize=10, fontweight='bold')
    ax.set_yticklabels(["Ambulans", "Damkar", "Polisi"], fontsize=10, fontweight='bold')
    ax.set_xlabel('Tebakan LCD Alat (Manual)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Suara Sebenarnya (Audio)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('CONFUSION MATRIX PENGUJIAN HARDWARE', fontsize=12, fontweight='bold', pad=25)

    for i in range(len(kelas_list)):
        for j in range(len(cols_display)):
            val = matrix_np[i, j]
            is_correct = (j < 3 and kelas_list[i] == cols_display[j])
            txt = f"{val}\n(✓)" if (is_correct and val > 0) else f"{val}"
            color = "white" if val > (matrix_np.max() / 2) else "black"
            ax.text(j, i, txt, ha='center', va='center', fontsize=12, fontweight='bold', color=color)

    plt.tight_layout()
    cm_path = os.path.join(target_grafik_dir, "grafik_confusion_matrix.png")
    fig.savefig(cm_path)
    plt.close(fig)
    chart_paths['confusion_matrix'] = cm_path

    # -------------------------------------------------------------
    # 3. DONUT CHART DISTRIBUSI HASIL (BENAR VS SALAH VS MISSED)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    total_benar = sum(confusion.get(k, {}).get(k, 0) for k in kelas_list)
    total_salah = sum(mode_count.get(k, {}).get("SALAH_DETEKSI", 0) for k in kelas_list)
    total_diam = sum(mode_count.get(k, {}).get("TIDAK_TERDETEKSI", 0) for k in kelas_list)
    total_lambat = sum(mode_count.get(k, {}).get("LAMBAT", 0) for k in kelas_list)

    labels = []
    sizes = []
    colors_pie = []

    if total_benar > 0:
        labels.append(f'Benar ({total_benar})')
        sizes.append(total_benar)
        colors_pie.append('#28A745')
    if total_lambat > 0:
        labels.append(f'Lambat ({total_lambat})')
        sizes.append(total_lambat)
        colors_pie.append('#17A2B8')
    if total_diam > 0:
        labels.append(f'Tidak Terdeteksi ({total_diam})')
        sizes.append(total_diam)
        colors_pie.append('#FFC107')
    if total_salah > 0:
        labels.append(f'Salah Deteksi ({total_salah})')
        sizes.append(total_salah)
        colors_pie.append('#DC3545')

    if sum(sizes) == 0:
        sizes = [1]
        labels = ['Belum ada data']
        colors_pie = ['#CCCCCC']

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=140, colors=colors_pie,
        wedgeprops=dict(width=0.42, edgecolor='w', linewidth=2),
        textprops=dict(fontsize=10, fontweight='bold')
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
        at.set_fontsize(11)

    ax.set_title('PROPORSI HASIL PENGUJIAN HARDWARE', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    pie_path = os.path.join(target_grafik_dir, "grafik_distribusi_performa.png")
    fig.savefig(pie_path)
    plt.close(fig)
    chart_paths['donut_distribusi'] = pie_path

    # -------------------------------------------------------------
    # 4. DASHBOARD GABUNGAN 3-IN-1 (UNTUK SKRIPSI/LAPORAN)
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(14, 8), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # Subplot 1: Bar chart akurasi
    ax1 = fig.add_subplot(gs[0, 0])
    b1 = ax1.bar(categories, scores, color=colors, width=0.5, edgecolor='#222', zorder=3)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel('Akurasi (%)', fontweight='bold')
    ax1.set_title('(A) Akurasi Deteksi per Kelas Sirine', fontweight='bold', fontsize=11)
    ax1.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)
    for b, s in zip(b1, scores):
        ax1.annotate(f'{s:.1f}%', xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    # Subplot 2: Donut
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors_pie,
            wedgeprops=dict(width=0.45, edgecolor='w', linewidth=2),
            textprops=dict(fontsize=9, fontweight='bold'))
    ax2.set_title('(B) Proporsi Status Evaluasi Alat', fontweight='bold', fontsize=11)

    # Subplot 3: Confusion matrix (Full row bottom)
    ax3 = fig.add_subplot(gs[1, :])
    cax3 = ax3.matshow(matrix_np, cmap='Blues', alpha=0.85)
    fig.colorbar(cax3, ax=ax3, fraction=0.03, pad=0.04)
    ax3.set_xticks(range(len(cols_display)))
    ax3.set_yticks(range(len(kelas_list)))
    ax3.set_xticklabels(["Ambulans", "Damkar", "Polisi", "Diam (0)"], fontsize=10, fontweight='bold')
    ax3.set_yticklabels(["Ambulans", "Damkar", "Polisi"], fontsize=10, fontweight='bold')
    ax3.set_xlabel('Prediksi LCD Alat (Manual)', fontweight='bold')
    ax3.set_ylabel('Ground Truth (Suara Asli)', fontweight='bold')
    ax3.set_title('(C) Confusion Matrix Evaluasi Deteksi', fontweight='bold', fontsize=11, pad=15)
    for i in range(len(kelas_list)):
        for j in range(len(cols_display)):
            val = matrix_np[i, j]
            is_correct = (j < 3 and kelas_list[i] == cols_display[j])
            txt = f"{val}\n(✓)" if (is_correct and val > 0) else f"{val}"
            color = "white" if val > (matrix_np.max() / 2) else "black"
            ax3.text(j, i, txt, ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    plt.suptitle(f'DASHBOARD DATA ANALYSIS SIRENMASTER (TOTAL AKURASI: {overall_accuracy:.1f}%)',
                 fontsize=14, fontweight='bold', y=0.98)
    dashboard_path = os.path.join(target_grafik_dir, "dashboard_analisis_lengkap.png")
    fig.savefig(dashboard_path)
    plt.close(fig)
    chart_paths['dashboard_lengkap'] = dashboard_path

    return chart_paths


def export_benchmark_to_excel_and_charts(results_log, confusion, mode_count, duration_sec, folder_name=None):
    """
    Ekspor lengkap hasil benchmark ke folder per pengujian (Pengujian_1, Pengujian_2, ...)
    berisi:
    - Dashboard_Analisis_Pengujian_X.xlsx
    - Ringkasan_Analisis_Pengujian_X.txt
    - grafik/ (4 file grafik 300 DPI)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not folder_name:
        _, folder_name = get_next_pengujian_name(OUTPUT_DIR)

    session_dir = os.path.join(OUTPUT_DIR, folder_name)
    session_grafik_dir = os.path.join(session_dir, "grafik")
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(session_grafik_dir, exist_ok=True)

    timestamp_file = time.strftime("%Y%m%d_%H%M%S")
    timestamp_readable = time.strftime("%d %B %Y, %H:%M:%S")

    KELAS = ["AMBULANCE", "DAMKAR", "POLISI"]

    # Hitung metrik keseluruhan
    total_valid = sum(sum(c.values()) for c in confusion.values())
    total_benar = sum(confusion.get(k, {}).get(k, 0) for k in KELAS)
    acc_total = (total_benar / total_valid * 100) if total_valid > 0 else 0
    total_salah = sum(mode_count.get(k, {}).get("SALAH_DETEKSI", 0) for k in KELAS)
    total_diam = sum(mode_count.get(k, {}).get("TIDAK_TERDETEKSI", 0) for k in KELAS)
    total_lambat = sum(mode_count.get(k, {}).get("LAMBAT", 0) for k in KELAS)

    # Buat grafik visualisasi ke folder grafik sesi ini
    chart_paths = generate_charts(confusion, mode_count, acc_total, timestamp_file, session_grafik_dir)

    # Inisialisasi Workbook Excel
    wb = openpyxl.Workbook()

    # Styling Palettes
    font_title = Font(name="Segoe UI", size=15, bold=True, color="FFFFFF")
    font_subtitle = Font(name="Segoe UI", size=11, italic=True, color="FFFFFF")
    font_section = Font(name="Segoe UI", size=12, bold=True, color="1B365D")
    font_header = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    font_bold = Font(name="Segoe UI", size=10, bold=True)
    font_regular = Font(name="Segoe UI", size=10)

    fill_title = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    fill_header = PatternFill(start_color="2E86AB", end_color="2E86AB", fill_type="solid")
    fill_header_dark = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")

    fill_kpi_card = PatternFill(start_color="F2F4F8", end_color="F2F4F8", fill_type="solid")
    fill_benar = PatternFill(start_color="D4EDDA", end_color="D4EDDA", fill_type="solid")
    fill_salah = PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid")
    fill_diam = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")
    fill_lambat = PatternFill(start_color="D1ECF1", end_color="D1ECF1", fill_type="solid")

    font_benar = Font(name="Segoe UI", size=10, bold=True, color="155724")
    font_salah = Font(name="Segoe UI", size=10, bold=True, color="721C24")
    font_diam = Font(name="Segoe UI", size=10, bold=True, color="856404")
    font_lambat = Font(name="Segoe UI", size=10, bold=True, color="0C5460")

    thin_border = Border(
        left=Side(style='thin', color='C8D1DC'),
        right=Side(style='thin', color='C8D1DC'),
        top=Side(style='thin', color='C8D1DC'),
        bottom=Side(style='thin', color='C8D1DC')
    )

    align_center = Alignment(horizontal='center', vertical='center')
    align_left = Alignment(horizontal='left', vertical='center')
    align_right = Alignment(horizontal='right', vertical='center')

    # =========================================================================
    # SHEET 1: EXECUTIVE DASHBOARD DATA ANALYSIS
    # =========================================================================
    ws1 = wb.active
    ws1.title = "Executive Dashboard"
    ws1.views.sheetView[0].showGridLines = True

    # Title Banner
    ws1.merge_cells("A1:H2")
    ws1["A1"] = f"DASHBOARD DATA ANALYSIS — {folder_name.upper()}"
    ws1["A1"].font = font_title
    ws1["A1"].fill = fill_title
    ws1["A1"].alignment = align_center

    # Metadata Info Box (Baris 4-7)
    ws1["A4"] = "Sesi Pengujian:"
    ws1["B4"] = folder_name
    ws1["A5"] = "Waktu Uji:"
    ws1["B5"] = timestamp_readable
    ws1["A6"] = "Durasi:"
    ws1["B6"] = f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s"
    ws1["A7"] = "Status Model:"
    ws1["B7"] = "LULUS SKRIPSI (>=90%)" if acc_total >= 90 else "PERLU RETRAIN (<90%)"

    for r in range(4, 8):
        ws1[f"A{r}"].font = font_bold
        ws1[f"B{r}"].font = font_regular
        ws1[f"B{r}"].alignment = align_left

    # KPI Summary Cards (C4:H5)
    cards = [
        ("D", "AKURASI TOTAL", f"{acc_total:.1f}%", fill_benar if acc_total >= 85 else fill_salah, "155724" if acc_total >= 85 else "721C24"),
        ("E", "BENAR", f"{total_benar}", fill_kpi_card, "155724"),
        ("F", "SALAH KELAS", f"{total_salah}", fill_kpi_card, "721C24"),
        ("G", "MISSED (0)", f"{total_diam}", fill_kpi_card, "856404"),
        ("H", "LAMBAT (~)", f"{total_lambat}", fill_kpi_card, "0C5460")
    ]
    for col_letter, title_kpi, val_kpi, fill_card, font_col in cards:
        ws1[f"{col_letter}4"] = title_kpi
        ws1[f"{col_letter}4"].font = font_header
        ws1[f"{col_letter}4"].fill = fill_header_dark
        ws1[f"{col_letter}4"].alignment = align_center

        ws1[f"{col_letter}5"] = val_kpi
        ws1[f"{col_letter}5"].font = Font(name="Segoe UI", size=15, bold=True, color=font_col)
        ws1[f"{col_letter}5"].fill = fill_card
        ws1[f"{col_letter}5"].alignment = align_center
        ws1[f"{col_letter}5"].border = thin_border

    # Tabel 1: Evaluasi Performa per Kelas (Baris 9+)
    ws1["A9"] = "1. TABEL EVALUASI PER KELAS SIRINE (STATISTIK AKURASI)"
    ws1["A9"].font = font_section

    headers_k = ["Kategori Sirine", "Sampel Uji", "Benar", "Salah Deteksi", "Tidak Deteksi", "Lambat", "Akurasi (%)", "Target", "Status Kelulusan"]
    for col_idx, h in enumerate(headers_k, start=1):
        cell = ws1.cell(row=10, column=col_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = align_center
        cell.border = thin_border

    row_k = 11
    for k in KELAS:
        tot_k = sum(confusion.get(k, {}).values())
        bk = confusion.get(k, {}).get(k, 0)
        sk = mode_count.get(k, {}).get("SALAH_DETEKSI", 0)
        dk = mode_count.get(k, {}).get("TIDAK_TERDETEKSI", 0)
        lk = mode_count.get(k, {}).get("LAMBAT", 0)
        acc_k = (bk / tot_k * 100) if tot_k else 0
        lulus_txt = "LULUS (>=90%) ✓" if acc_k >= 90 else "EVALUASI (<90%) ⚠"

        vals = [k, tot_k, bk, sk, dk, lk, f"{acc_k:.1f}%", "90.0%", lulus_txt]
        for col_idx, val in enumerate(vals, start=1):
            cell = ws1.cell(row=row_k, column=col_idx, value=val)
            cell.font = font_regular
            cell.border = thin_border
            cell.alignment = align_center if col_idx > 1 else align_left
            if col_idx == 9:
                cell.font = font_benar if acc_k >= 90 else font_salah

        row_k += 1

    # Tabel 2: Confusion Matrix (Baris row_k + 2)
    row_cm = row_k + 2
    ws1.cell(row=row_cm, column=1, value="2. CONFUSION MATRIX DETEKSI (BARIS = GROUND TRUTH | KOLOM = PREDIKSI LCD)").font = font_section

    cm_headers = ["Kelas Sebenarnya", "Prediksi Ambulans", "Prediksi Damkar", "Prediksi Polisi", "Tidak Bereaksi (0)"]
    for col_idx, h in enumerate(cm_headers, start=1):
        cell = ws1.cell(row=row_cm + 1, column=col_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = align_center
        cell.border = thin_border

    for i, k in enumerate(KELAS, start=row_cm + 2):
        row_vals = [
            k,
            confusion.get(k, {}).get("AMBULANCE", 0),
            confusion.get(k, {}).get("DAMKAR", 0),
            confusion.get(k, {}).get("POLISI", 0),
            confusion.get(k, {}).get("TDK_DETEKSI", 0)
        ]
        for col_idx, val in enumerate(row_vals, start=1):
            cell = ws1.cell(row=i, column=col_idx, value=val)
            cell.border = thin_border
            cell.alignment = align_center if col_idx > 1 else align_left
            cell.font = font_regular
            # Highlight diagonal cells (Benar)
            if col_idx == (KELAS.index(k) + 2):
                cell.fill = fill_benar
                cell.font = font_benar

    # Atur lebar kolom Sheet 1
    for col in ws1.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max(len(str(cell.value or '')) for cell in col)
        ws1.column_dimensions[col_letter].width = max(max_len + 4, 13)

    # =========================================================================
    # SHEET 2: DATA ANALYSIS (DETAIL 100 SUARA)
    # =========================================================================
    ws2 = wb.create_sheet(title="Data Analysis Detail")
    ws2.views.sheetView[0].showGridLines = True

    ws2.merge_cells("A1:H1")
    ws2["A1"] = f"DATA MENTAH & ANALISIS PENGUJIAN PER SUARA — {folder_name.upper()}"
    ws2["A1"].font = font_title
    ws2["A1"].fill = fill_title
    ws2["A1"].alignment = align_center

    headers_raw = ["ID / Ronde", "Timestamp", "Nama File Audio (.wav)", "Ground Truth", "Input LCD Manual", "Status Evaluasi", "Skor (1/0)", "Catatan Analisis"]
    for col_idx, h in enumerate(headers_raw, start=1):
        cell = ws2.cell(row=3, column=col_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = align_center
        cell.border = thin_border

    for r_idx, item in enumerate(results_log, start=4):
        if isinstance(item, dict):
            ronde = item.get("ronde", r_idx - 3)
            waktu = item.get("waktu", "-")
            fname = item.get("file", "-")
            k_asli = item.get("kelas_asli", "-")
            tebakan = item.get("tebakan", "-")
            mode = item.get("mode", "-")
            skor = item.get("skor", 1 if mode in ("BENAR", "LAMBAT") else 0)
            ket = item.get("keterangan", "")
        else:
            ronde, k_asli, tebakan, mode = item[:4]
            fname = item[4] if len(item) > 4 else "-"
            waktu = item[5] if len(item) > 5 else "-"
            skor = 1 if mode in ("BENAR", "LAMBAT") else 0
            ket = "Tepat sesuai suara" if mode == "BENAR" else ("Miskalsifikasi" if mode == "SALAH_DETEKSI" else "Tidak bereaksi")

        if mode == "BENAR":
            status_text = "BENAR ✓"
            fill_curr = fill_benar
            font_curr = font_benar
        elif mode == "SALAH_DETEKSI":
            status_text = "SALAH DETEKSI ✗"
            fill_curr = fill_salah
            font_curr = font_salah
        elif mode == "TIDAK_TERDETEKSI":
            status_text = "TIDAK TERDETEKSI 0"
            fill_curr = fill_diam
            font_curr = font_diam
        elif mode == "LAMBAT":
            status_text = "BENAR (LAMBAT) ~"
            fill_curr = fill_lambat
            font_curr = font_lambat
        else:
            status_text = mode
            fill_curr = None
            font_curr = font_regular

        row_data = [ronde, waktu, fname, k_asli, tebakan, status_text, skor, ket]
        for col_idx, val in enumerate(row_data, start=1):
            cell = ws2.cell(row=r_idx, column=col_idx, value=val)
            cell.font = font_regular
            cell.border = thin_border
            if col_idx in (1, 2, 4, 5, 7):
                cell.alignment = align_center
            elif col_idx == 6:
                cell.alignment = align_center
                cell.font = font_curr
                if fill_curr:
                    cell.fill = fill_curr
            else:
                cell.alignment = align_left

    for col in ws2.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max(len(str(cell.value or '')) for cell in col)
        ws2.column_dimensions[col_letter].width = max(max_len + 3, 11)

    ws2.auto_filter.ref = f"A3:H{len(results_log) + 3}"

    # =========================================================================
    # SHEET 3: GRAFIK & VISUALISASI SKRIPSI
    # =========================================================================
    ws3 = wb.create_sheet(title="Grafik & Visualisasi")
    ws3.views.sheetView[0].showGridLines = True

    ws3.merge_cells("A1:K1")
    ws3["A1"] = f"VISUALISASI DATA ANALYSIS — {folder_name.upper()}"
    ws3["A1"].font = font_title
    ws3["A1"].fill = fill_title
    ws3["A1"].alignment = align_center

    ws3["A3"] = "Grafik ini otomatis di-generate beresolusi 300 DPI untuk Bab 4 Skripsi."
    ws3["A3"].font = font_bold

    # Sisipkan Gambar Grafik Dashboard Lengkap
    if os.path.exists(chart_paths.get('dashboard_lengkap', '')):
        try:
            img_dash = OpenpyxlImage(chart_paths['dashboard_lengkap'])
            img_dash.width = 750
            img_dash.height = 430
            ws3.add_image(img_dash, "A5")
        except Exception as e:
            ws3["A5"] = f"[!] Gagal menyematkan grafik: {e}"

    # Simpan File Excel di folder sesi Pengujian_X
    excel_name = f"Dashboard_Analisis_{folder_name}.xlsx"
    excel_path = os.path.join(session_dir, excel_name)
    excel_latest = os.path.join(OUTPUT_DIR, "Dashboard_Analisis_Terbaru.xlsx")

    wb.save(excel_path)
    wb.save(excel_latest)

    # Simpan Laporan Naratif Teks (Ringkasan_Analisis_Pengujian_X.txt)
    txt_path = os.path.join(session_dir, f"Ringkasan_Analisis_{folder_name}.txt")
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write(f"RINGKASAN DATA ANALYSIS — {folder_name.upper()}\n")
            f.write(f"SISTEM DETEKSI SIRENE DARURAT SIRENMASTER\n")
            f.write("="*70 + "\n\n")
            f.write(f"Tanggal Pengujian : {timestamp_readable}\n")
            f.write(f"Durasi Pengujian  : {int(duration_sec // 60)} Menit {int(duration_sec % 60)} Detik\n")
            f.write(f"Total Sampel Uji  : {len(results_log)} Suara\n")
            f.write(f"Akurasi Rata-rata : {acc_total:.2f}%\n\n")
            f.write("HASIL DETEKSI KESELURUHAN:\n")
            f.write(f" - Benar Sesuai Kelas : {total_benar} suara ({acc_total:.1f}%)\n")
            f.write(f" - Salah Deteksi      : {total_salah} suara\n")
            f.write(f" - Tidak Terdeteksi   : {total_diam} suara\n")
            f.write(f" - Lambat Respon      : {total_lambat} suara\n\n")
            f.write("PERFORMA PER KELAS:\n")
            for k in KELAS:
                tot_k = sum(confusion.get(k, {}).values())
                bk = confusion.get(k, {}).get(k, 0)
                acc_k = (bk / tot_k * 100) if tot_k else 0
                f.write(f" • {k:<12} : {bk}/{tot_k} ({acc_k:.1f}%) -> {'LULUS' if acc_k >= 90 else 'PERLU EVALUASI'}\n")
            f.write("\nFile Excel dan grafik resolusi 300 DPI tersimpan di folder ini.\n")
    except Exception:
        pass

    return {
        "folder_name": folder_name,
        "session_dir": session_dir,
        "excel_path": excel_path,
        "excel_latest": excel_latest,
        "chart_paths": chart_paths,
        "txt_path": txt_path,
        "output_dir": OUTPUT_DIR
    }
