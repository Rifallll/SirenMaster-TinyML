import os
import sys
import datetime

# Set console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import docx
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls
except ImportError:
    print("[!] Modul python-docx belum terinstal.")
    sys.exit(1)

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def add_code_diff_block(doc, title, before_code, after_code, explanation):
    h = doc.add_heading(title, level=2)
    h.style.font.name = "Arial"
    h.style.font.size = Pt(11.5)
    h.style.font.color.rgb = RGBColor(30, 60, 114)

    p_exp = doc.add_paragraph(explanation)
    p_exp.style.font.name = "Arial"
    p_exp.style.font.size = Pt(9.5)

    tbl = doc.add_table(rows=2, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    h_before = tbl.rows[0].cells[0]
    h_after = tbl.rows[0].cells[1]
    h_before.width = Inches(3.3)
    h_after.width = Inches(3.4)
    
    set_cell_background(h_before, "FFEBEE")
    set_cell_background(h_after, "E8F5E9")
    
    p_hb = h_before.paragraphs[0]
    r_hb = p_hb.add_run("❌ SEBELUM (KODE LAMA / MASALAH)")
    r_hb.font.name = "Arial"
    r_hb.font.size = Pt(8.5)
    r_hb.font.bold = True
    r_hb.font.color.rgb = RGBColor(180, 0, 0)

    p_ha = h_after.paragraphs[0]
    r_ha = p_ha.add_run("✅ SESUDAH (KODE BARU / OPTIMAL)")
    r_ha.font.name = "Arial"
    r_ha.font.size = Pt(8.5)
    r_ha.font.bold = True
    r_ha.font.color.rgb = RGBColor(0, 120, 40)

    c_before = tbl.rows[1].cells[0]
    c_after = tbl.rows[1].cells[1]
    set_cell_background(c_before, "FFF9F9")
    set_cell_background(c_after, "F6FFF6")

    p_cb = c_before.paragraphs[0]
    r_cb = p_cb.add_run(before_code.strip())
    r_cb.font.name = "Consolas"
    r_cb.font.size = Pt(8.0)
    r_cb.font.color.rgb = RGBColor(80, 20, 20)

    p_ca = c_after.paragraphs[0]
    r_ca = p_ca.add_run(after_code.strip())
    r_ca.font.name = "Consolas"
    r_ca.font.size = Pt(8.0)
    r_ca.font.color.rgb = RGBColor(10, 70, 20)

    doc.add_paragraph()

def create_word_report(output_path="LAPORAN_MODEL_HARIAN.docx"):
    doc = docx.Document()

    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    today_str = datetime.date.today().strftime("%d %B %Y")
    branch_name = "model/2026-10-02-v1"

    # --- TITLE & HEADER ---
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("LAPORAN EVALUASI & BEDAH DATA MODEL AI SIRENMASTER")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(16)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(16, 44, 87)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("Dokumentasi Riil Hasil Pengujian (Validasi Data vs Live Hardware) & Analisis Salah Kamar")
    run_sub.font.name = "Arial"
    run_sub.font.size = Pt(10.5)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(80, 80, 80)

    doc.add_paragraph()

    # --- INFO BOX ---
    info_table = doc.add_table(rows=6, cols=2)
    info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    info_data = [
        ("Tanggal Laporan", today_str),
        ("Git Branch", branch_name),
        ("Target Board", "WeMos LOLIN S2 Mini (ESP32-S2FN4R2, 4MB Flash, 2MB PSRAM)"),
        ("Akurasi Dataset Validasi", "79.16% (Data Riil evaluation_report.csv)"),
        ("Akurasi Pengujian Live 100 Sirine", "57.00% (57 Benar, 32 Diam/Tidak Memicu, 11 Salah Kamar)"),
        ("Status Kompilasi Arduino IDE", "PASSED (Flash: 16%, SRAM: 16%, 0 Error)")
    ]

    for i, (label, val) in enumerate(info_data):
        row = info_table.rows[i]
        c1, c2 = row.cells[0], row.cells[1]
        c1.width = Inches(2.3)
        c2.width = Inches(4.4)
        set_cell_background(c1, "F0F4F8")
        set_cell_background(c2, "FFFFFF")
        
        p1 = c1.paragraphs[0]
        r1 = p1.add_run(label)
        r1.font.bold = True
        r1.font.name = "Arial"
        r1.font.size = Pt(9.0)
        
        p2 = c2.paragraphs[0]
        r2 = p2.add_run(val)
        r2.font.name = "Arial"
        r2.font.size = Pt(9.0)
        if "Akurasi" in label:
            r2.font.bold = True
        if "Arduino" in label:
            r2.font.bold = True
            r2.font.color.rgb = RGBColor(0, 130, 50)

    doc.add_paragraph()

    # --- SECTION 1: DATA RIIL EVALUASI & DARI MANA F1-SCORE DIDAPAT ---
    h1 = doc.add_heading("1. Sumber Data Riil & Penjelasan Perhitungan F1-Score", level=1)
    h1.style.font.name = "Arial"
    h1.style.font.color.rgb = RGBColor(16, 44, 87)

    p_f1 = doc.add_paragraph(
        "F1-Score bukan angka rekaan, melainkan hasil perhitungan matematis standar machine learning "
        "yang dihitung dari pengujian dataset validasi (2.764 sampel audio independen pada file `evaluation_report.csv`).\n\n"
        "Rumus perhitungan yang digunakan adalah:"
    )
    p_f1.style.font.name = "Arial"
    p_f1.style.font.size = Pt(9.5)

    p_formula = doc.add_paragraph(
        "• Precision = True Positive / (True Positive + False Positive)  → Dari yang diprediksi kelas X, berapa yang aslinya memang X?\n"
        "• Recall    = True Positive / (True Positive + False Negative)  → Dari total suara asli kelas X, berapa persen yang berhasil tertangkap?\n"
        "• F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)   → Rata-rata harmonik antara ketepatan dan ketuntasan deteksi."
    )
    p_formula.style.font.name = "Consolas"
    p_formula.style.font.size = Pt(8.5)
    p_formula.paragraph_format.left_indent = Inches(0.2)

    # Tabel Data Riil CSV
    doc.add_paragraph().add_run("Tabel 1: Hasil Evaluasi Dataset Validasi Riil (2.764 Sampel Audio)").bold = True
    csv_table = doc.add_table(rows=6, cols=5)
    csv_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    csv_headers = ["Kategori Suara", "Precision", "Recall", "F1-Score", "Jumlah Sampel (Support)"]

    for j, h_text in enumerate(csv_headers):
        cell = csv_table.rows[0].cells[j]
        set_cell_background(cell, "102C57")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_text)
        r.font.name = "Arial"
        r.font.size = Pt(9.0)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    csv_rows = [
        ("AMBULANCE", "64.5%", "82.8%", "72.5%", "145 file"),
        ("FIRETRUCK (Damkar)", "93.9%", "66.7%", "78.0%", "1.328 file"),
        ("POLICE (Polisi)", "68.7%", "91.6%", "78.5%", "939 file"),
        ("NORMAL (Kebisingan)", "84.1%", "91.5%", "87.6%", "352 file"),
        ("RATA-RATA TOTAL (Accuracy)", "79.2%", "79.2%", "79.2%", "2.764 file")
    ]

    for i, r_data in enumerate(csv_rows):
        row = csv_table.rows[i+1]
        is_last = (i == len(csv_rows) - 1)
        bg = "EBF2FA" if is_last else ("F9FAFB" if i % 2 == 1 else "FFFFFF")
        for j, val in enumerate(r_data):
            cell = row.cells[j]
            set_cell_background(cell, bg)
            p = cell.paragraphs[0]
            if j > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.name = "Arial"
            r.font.size = Pt(9.0)
            if is_last or j == 0:
                r.font.bold = True

    doc.add_paragraph()

    # --- SECTION 2: BEDAH MASALAH BENCHMARK 100 SIRINE (SALAH KAMAR & DIAM) ---
    h2 = doc.add_heading("2. Bedah Masalah Pengujian Nyata: Mengapa Terjadi 'Salah Kamar' dan 'Diam'?", level=1)
    h2.style.font.name = "Arial"
    h2.style.font.color.rgb = RGBColor(16, 44, 87)

    p_bm_intro = doc.add_paragraph(
        "Pada pengujian live 100 audio (`hasil_benchmark_100.txt`), akurasi riil yang didapatkan adalah 57.0%. "
        "Berikut rincian fakta lapangan penyebab kegagalan dan solusinya:"
    )
    p_bm_intro.style.font.name = "Arial"
    p_bm_intro.style.font.size = Pt(9.5)

    bm_table = doc.add_table(rows=5, cols=4)
    bm_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    bm_headers = ["Kategori Pengujian", "Total Suara", "Berhasil Benar", "Akurasi Nyata"]

    for j, h_text in enumerate(bm_headers):
        cell = bm_table.rows[0].cells[j]
        set_cell_background(cell, "102C57")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_text)
        r.font.name = "Arial"
        r.font.size = Pt(9.0)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    bm_rows = [
        ("AMBULANCE", "33 audio", "31 audio", "93.9% (Sangat Baik)"),
        ("DAMKAR (FIRETRUCK)", "33 audio", "4 audio", "12.1% (Penyumbang Error Terbesar)"),
        ("POLISI (POLICE)", "34 audio", "22 audio", "64.7% (Cukup Baik)"),
        ("TOTAL LIVE BENCHMARK", "100 audio", "57 audio", "57.0% (57 Benar, 32 Diam, 11 Salah Kamar)")
    ]

    for i, r_data in enumerate(bm_rows):
        row = bm_table.rows[i+1]
        is_last = (i == len(bm_rows) - 1)
        bg = "EBF2FA" if is_last else ("F9FAFB" if i % 2 == 1 else "FFFFFF")
        for j, val in enumerate(r_data):
            cell = row.cells[j]
            set_cell_background(cell, bg)
            p = cell.paragraphs[0]
            if j > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.name = "Arial"
            r.font.size = Pt(9.0)
            if is_last or j == 0:
                r.font.bold = True
            if "12.1%" in val:
                r.font.color.rgb = RGBColor(180, 0, 0)

    doc.add_paragraph()

    # Analisis Penyebab
    p_why = doc.add_paragraph()
    p_why.add_run("FAKTOR PENYEBAB DAN SOLUSINYA:\n").bold = True
    
    reasons = [
        ("A. Mengapa Ada 32 Suara 'DIAM' (Tidak Memicu Siren)?",
         "Penyebab: Ambang batas pemicu sirene pada firmware C++ diset ketat (bestSirenScore >= 0.58f dan cs[bestSiren] >= 0.60f). "
         "Jika suara sirene yang diputar volumenya rendah, atau nada ayunannya sedang berada di frekuensi lembah, skor sirine hanya mencapai 40%-50%, "
         "sehingga firmware menganggapnya sebagai Normal. Sistem sengaja memilih 'diam' agar tidak terjadi false alarm sembarangan."),
        
        ("B. Mengapa Terjadi 11 Suara 'SALAH KAMAR' (Terutama Damkar 12.1%)?",
         "Penyebab: Damkar di Indonesia menggunakan kombinasi nada Yelp (cepat) dan Horn/Phaser. Nada Yelp damkar memiliki spektrum frekuensi yang tumpang tindih "
         "dengan nada Polisi dan Ambulans. Karena data latih Polisi/Ambulans memiliki sampel yang lebih kaya, model AI cenderung 'salah kamar' mengklasifikasikan "
         "Damkar sebagai Polisi atau Ambulans ketika ragu."),
        
        ("C. Solusi yang Sedang Diterapkan Hari Ini:",
         "1. Menjalankan `fix_firetruck_augment.py` untuk melipatgandakan data Damkar dengan variasi pitch shift, time-stretch, dan noise injection.\n"
         "2. Menyeimbangkan kalibrasi bobot `CAL[NUM_CLASSES]` di firmware ESP32 agar Damkar tidak tertelan oleh Ambulans/Polisi.")
    ]

    for t, d in reasons:
        p_r = doc.add_paragraph()
        r_t = p_r.add_run(f"• {t}\n")
        r_t.bold = True
        r_t.font.color.rgb = RGBColor(30, 60, 114)
        p_r.add_run(d)
        p_r.paragraph_format.left_indent = Inches(0.15)

    doc.add_paragraph()

    # --- SECTION 3: PERBANDINGAN KODE BEFORE VS AFTER ---
    h3 = doc.add_heading("3. Bukti Perubahan Kode Program (Before vs After)", level=1)
    h3.style.font.name = "Arial"
    h3.style.font.color.rgb = RGBColor(16, 44, 87)

    add_code_diff_block(
        doc,
        title="A. Perubahan Ambang Batas Pemicu Sirene (Mengurangi False Alarm vs Sensitivitas)",
        before_code="""// Ambang batas lama (Longgar):
bool sirenTrigger = 
  (bestSirenScore >= 0.40f && bestSirenScore > cs[2]) ||
  (totalSiren >= 0.50f && bestSirenScore >= 0.32f);

// Akibat: Sangat sensitif, tapi suara
// klakson / orang ngobrol bisa memicu.""",
        after_code="""// Ambang batas baru (Ketat Anti False-Alarm):
bool sirenTrigger = 
  (bestSirenScore >= 0.58f && 
   cs[bestSiren] >= 0.60f && 
   bestSirenScore > cs[2]);

// Hasil: Kebal suara ruangan bising,
// namun butuh volume sirine cukup jelas.""",
        explanation="Perubahan ini menjelaskan mengapa pada pengujian ada audio yang 'diam'. Ambang batas dinaikkan ke 60% untuk mengorbankan sirine yang sangat sayup-sayup demi memastikan di jalan raya tidak terjadi salah deteksi saat tidak ada ambulans."
    )

    add_code_diff_block(
        doc,
        title="B. Algoritma Blended Consensus (Pencegah Salah Kamar 1-Frame)",
        before_code="""// Mengambil keputusan dari 1 frame instan:
int bestSiren = 0;
float bestSirenScore = cs[0];
if (cs[1] > bestSirenScore) bestSiren = 1;
if (cs[3] > bestSirenScore) bestSiren = 3;""",
        after_code="""// Menggabungkan 50% skor instan + 50% EMA:
float blended[NUM_CLASSES];
for (int i = 0; i < NUM_CLASSES; i++) {
  blended[i] = 0.5f * cs[i] + 0.5f * ema_probs[i];
}
// Mencegah spike 1-frame acak mengunci salah kamar""",
        explanation="Dengan algoritma Blended Consensus, fluktuasi sesaat dari nada sirine tidak langsung mengubah kelas sirine, melainkan harus konsisten selama beberapa frame berturut-turut."
    )

    doc.add_paragraph()

    # --- SECTION 4: KESIMPULAN & STATUS UPLOAD KAMIS/JUMAT ---
    h4 = doc.add_heading("4. Status Kesiapan Upload Hardware (Kamis & Jumat)", level=1)
    h4.style.font.name = "Arial"
    h4.style.font.color.rgb = RGBColor(16, 44, 87)

    p_status = doc.add_paragraph(
        "• Kompilasi Arduino IDE: PASSED (Flash 514 KB / 3.14 MB [16%], SRAM 54 KB / 327 KB [16%])\n"
        "• Rekomendasi Pengujian: Putar audio pada volume sedang-tinggi (jarak 0.5 - 2 meter dari mikrofon ESP32 INMP441) agar sinyal melampaui ambang batas 60%.\n"
        "• Fokus Retraining Berikutnya: Meningkatkan recall Damkar melalui penambahan variasi nada Phaser & Horn pada dataset."
    )
    p_status.style.font.name = "Arial"
    p_status.style.font.size = Pt(9.5)

    doc.save(output_path)
    print(f"[+] File Word laporan riil berhasil diperbarui: {output_path}")

if __name__ == "__main__":
    create_word_report()
