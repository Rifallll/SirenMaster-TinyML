import os
import sys
import glob
import json

ROOT = r"C:\Users\ASUS\Videos\DATASET"
TRASH_DIR = os.path.join(ROOT, "TRASH", "audit_hasil_pemisahan")

bukan_sirine_files = glob.glob(os.path.join(TRASH_DIR, "bukan_sirine", "*.*"))
salah_kamar_files = glob.glob(os.path.join(TRASH_DIR, "salah_kamar", "*.*"))
hening_files = glob.glob(os.path.join(TRASH_DIR, "hening_rusak", "*.*"))

def generate_html_reviewer():
    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SirenMaster — Panel Review File Dataset Anomali</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #38bdf8;
            margin-bottom: 8px;
        }}
        .subtitle {{
            color: #94a3b8;
            margin-bottom: 24px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .card {{
            background-color: #1e293b;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #334155;
        }}
        .card.warning {{ border-left: 5px solid #f59e0b; }}
        .card.danger {{ border-left: 5px solid #ef4444; }}
        .card.info {{ border-left: 5px solid #3b82f6; }}
        .card-num {{
            font-size: 32px;
            font-weight: bold;
            color: #f8fafc;
        }}
        .card-title {{
            color: #94a3b8;
            font-size: 14px;
            margin-top: 4px;
        }}
        .section-title {{
            font-size: 20px;
            font-weight: bold;
            color: #e2e8f0;
            margin-top: 32px;
            margin-bottom: 16px;
            border-bottom: 2px solid #334155;
            padding-bottom: 8px;
        }}
        .table-wrap {{
            overflow-x: auto;
            background-color: #1e293b;
            border-radius: 12px;
            border: 1px solid #334155;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 14px;
        }}
        th {{
            background-color: #0f172a;
            color: #38bdf8;
            padding: 12px 16px;
            font-weight: 600;
        }}
        td {{
            padding: 12px 16px;
            border-top: 1px solid #334155;
            color: #cbd5e1;
        }}
        tr:hover {{
            background-color: #243248;
        }}
        audio {{
            height: 32px;
            width: 240px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge-red {{ background: #7f1d1d; color: #fca5a5; }}
        .badge-yellow {{ background: #78350f; color: #fde68a; }}
        .badge-blue {{ background: #1e3a8a; color: #93c5fd; }}
        .search-box {{
            width: 100%;
            padding: 12px 16px;
            border-radius: 8px;
            background: #1e293b;
            border: 1px solid #334155;
            color: white;
            font-size: 15px;
            margin-bottom: 20px;
            box-sizing: border-box;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Panel Review File Dataset Anomali</h1>
        <div class="subtitle">SirenMaster TinyML — Periksa & Dengarkan File yang Terdeteksi Bukan Sirine / Salah Kamar</div>

        <div class="stats-grid">
            <div class="card warning">
                <div class="card-num">{len(bukan_sirine_files)}</div>
                <div class="card-title">Bukan Sirine (Noise Murni)</div>
            </div>
            <div class="card danger">
                <div class="card-num">{len(salah_kamar_files)}</div>
                <div class="card-title">Salah Kamar Dominan</div>
            </div>
            <div class="card info">
                <div class="card-num">{len(hening_files)}</div>
                <div class="card-title">Hening / Rusak</div>
            </div>
        </div>

        <input type="text" id="search" class="search-box" placeholder="Ketik nama file untuk mencari..." onkeyup="filterTable()">

        <!-- TABEL BUKAN SIRINE -->
        <div class="section-title">1. Suara Bukan Sirine di Folder Siren ({len(bukan_sirine_files)} File)</div>
        <div class="table-wrap">
            <table id="table-bukan-sirine">
                <thead>
                    <tr>
                        <th>No</th>
                        <th>Nama File</th>
                        <th>Kategori Folder Asal</th>
                        <th>Putar Audio</th>
                    </tr>
                </thead>
                <tbody>
"""
    for idx, fp in enumerate(bukan_sirine_files):
        fname = os.path.basename(fp)
        cat_badge = "badge-yellow"
        cat_name = fname.split('_')[0] if '_' in fname else 'UNKNOWN'
        html_content += f"""
                    <tr>
                        <td>{idx + 1}</td>
                        <td><strong>{fname}</strong></td>
                        <td><span class="badge {cat_badge}">{cat_name}</span></td>
                        <td>
                            <audio controls preload="none">
                                <source src="TRASH/audit_hasil_pemisahan/bukan_sirine/{fname}" type="audio/wav">
                            </audio>
                        </td>
                    </tr>
"""

    html_content += f"""
                </tbody>
            </table>
        </div>

        <!-- TABEL SALAH KAMAR -->
        <div class="section-title">2. Suara Salah Kamar ({len(salah_kamar_files)} File)</div>
        <div class="table-wrap">
            <table id="table-salah-kamar">
                <thead>
                    <tr>
                        <th>No</th>
                        <th>Nama File</th>
                        <th>Status Perpindahan</th>
                        <th>Putar Audio</th>
                    </tr>
                </thead>
                <tbody>
"""
    for idx, fp in enumerate(salah_kamar_files[:200]): # Show top 200 for fast browser rendering
        fname = os.path.basename(fp)
        status_label = fname.split('_')[0:4]
        status_text = " ".join(status_label)
        html_content += f"""
                    <tr>
                        <td>{idx + 1}</td>
                        <td><strong>{fname}</strong></td>
                        <td><span class="badge badge-red">{status_text}</span></td>
                        <td>
                            <audio controls preload="none">
                                <source src="TRASH/audit_hasil_pemisahan/salah_kamar/{fname}" type="audio/wav">
                            </audio>
                        </td>
                    </tr>
"""

    html_content += """
                </tbody>
            </table>
        </div>
    </div>

    <script>
        function filterTable() {
            var input = document.getElementById("search");
            var filter = input.value.toUpperCase();
            var tables = [document.getElementById("table-bukan-sirine"), document.getElementById("table-salah-kamar")];
            
            tables.forEach(table => {
                var tr = table.getElementsByTagName("tr");
                for (var i = 1; i < tr.length; i++) {
                    var td = tr[i].getElementsByTagName("td")[1];
                    if (td) {
                        var txtValue = td.textContent || td.innerText;
                        if (txtValue.toUpperCase().indexOf(filter) > -1) {
                            tr[i].style.display = "";
                        } else {
                            tr[i].style.display = "none";
                        }
                    }
                }
            });
        }
    </script>
</body>
</html>
"""

    output_html_path = os.path.join(ROOT, "review_dataset_anomali.html")
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] Panel Review HTML berhasil dibuat di: {output_html_path}")

if __name__ == "__main__":
    generate_html_reviewer()
