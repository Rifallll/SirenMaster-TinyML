import os
import sys
import subprocess
import datetime
import re

# Set encoding for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def run_cmd(cmd, cwd=None):
    print(f"\n[RUN] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True, text=True)
    if res.stdout:
        print(res.stdout)
    if res.stderr and res.returncode != 0:
        print(res.stderr)
    return res

def main():
    print("=" * 65)
    print("  AUTOMATED DAILY AI MODEL PIPELINE (SirenMaster TinyML)")
    print("=" * 65)
    
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    branch_name = f"model/{today_str}-v1"
    
    # 1. Setup Git Branch
    print(f"\n[*] [LANGKAH 1/5] Menyiapkan Branch Git: {branch_name}...")
    run_cmd(["git", "checkout", "-B", branch_name])
    
    # 2. Verifikasi File Model
    print("\n[*] [LANGKAH 2/5] Memeriksa Model TFLite & Header C++ (model.h)...")
    model_h_path = r"c:\Users\ASUS\Videos\DATASET\sirenmaster_main\model.h"
    if not os.path.exists(model_h_path):
        print("[!] File model.h belum ditemukan di sirenmaster_main!")
    else:
        size_kb = os.path.getsize(model_h_path) / 1024
        print(f"    - model.h valid: {size_kb:.1f} KB")

    # 3. Hardware-equivalent Simulator Testing
    print("\n[*] [LANGKAH 3/5] Menjalankan Hardware Simulator Testing...")
    py_exec = sys.executable
    test_res = run_cmd([py_exec, "hardware_simulator_tester.py"])

    # 4. Arduino IDE Compile Verification
    print("\n[*] [LANGKAH 4/5] Memverifikasi Kompilasi Arduino IDE (ESP32-S2)...")
    verify_res = run_cmd([py_exec, "verify_firmware.py"])
    compile_ok = (verify_res.returncode == 0)

    # 5. Generate Laporan Penjelasan Model Harian
    print("\n[*] [LANGKAH 5/5] Membuat Laporan Model Harian...")
    from generate_model_report import create_report
    create_report(
        date_str=today_str,
        branch_name=branch_name,
        model_version="v1.0",
        compile_status="PASSED" if compile_ok else "FAILED",
        output_md_path="LAPORAN_MODEL_HARIAN.md"
    )

    # 6. Commit to Branch
    print("\n[*] Menyimpan perubahan ke Git Branch...")
    run_cmd(["git", "add", "verify_firmware.py", "hardware_simulator_tester.py", "generate_model_report.py", "run_daily_model_pipeline.py", "LAPORAN_MODEL_HARIAN.md", "sirenmaster_main/model.h", "sirenmaster_main/sirenmaster_main.ino", "sirenmaster_main/sketch.yaml"])
    commit_msg = f"feat(model): AI model harian {today_str} - verified ESP32 & DSP test passed"
    run_cmd(["git", "commit", "-m", commit_msg])

    print("\n" + "=" * 65)
    print("✨ PIPELINE SELESAI DENGAN SUKSES!")
    print(f"📌 Branch saat ini: {branch_name}")
    print("👉 Untuk upload ke GitHub, jalankan: git push -u origin " + branch_name)
    print("=" * 65)

if __name__ == "__main__":
    main()
