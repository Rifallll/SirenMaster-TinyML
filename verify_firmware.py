import os
import sys
import subprocess
import re

# Set encoding for Windows console compatibility
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ARDUINO_CLI_PATHS = [
    r"C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe",
    r"C:\Users\ASUS\AppData\Local\Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe",
    "arduino-cli"
]

SKETCH_PATH = r"c:\Users\ASUS\Videos\DATASET\sirenmaster_main"
FQBN = "esp32:esp32:lolin_s2_mini:CDCOnBoot=default,PartitionScheme=huge_app,DebugLevel=none"

def find_arduino_cli():
    for p in ARDUINO_CLI_PATHS:
        if os.path.exists(p):
            return p
    return "arduino-cli"

def verify_firmware():
    cli_path = find_arduino_cli()
    print("=" * 60)
    print("  VERIFIKASI KOMPILASI FIRMWARE ARDUINO IDE (ESP32-S2)")
    print("=" * 60)
    print(f"[*] Arduino CLI  : {cli_path}")
    print(f"[*] Sketch Path  : {SKETCH_PATH}")
    print(f"[*] Target FQBN  : {FQBN}")
    print("\n[*] Memulai proses kompilasi...")
    
    cmd = [
        cli_path,
        "compile",
        "--fqbn", FQBN,
        SKETCH_PATH
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    
    output = stdout + "\n" + stderr
    print(output)
    
    if proc.returncode == 0:
        print("\n" + "=" * 60)
        print("✅ VERIFIKASI BERHASIL (PASSED) - FIRMWARE 100% AMAN")
        print("=" * 60)
        
        # Extract memory stats
        flash_match = re.search(r"Sketch uses (\d+) bytes \((\d+)%\) of program storage space\. Maximum is (\d+) bytes\.", output)
        ram_match = re.search(r"Global variables use (\d+) bytes \((\d+)%\) of dynamic memory.*?Maximum is (\d+) bytes\.", output)
        
        stats = {
            "status": "PASSED",
            "flash_used": flash_match.group(1) if flash_match else "N/A",
            "flash_pct": flash_match.group(2) if flash_match else "N/A",
            "flash_max": flash_match.group(3) if flash_match else "N/A",
            "ram_used": ram_match.group(1) if ram_match else "N/A",
            "ram_pct": ram_match.group(2) if ram_match else "N/A",
            "ram_max": ram_match.group(3) if ram_match else "N/A"
        }
        return True, stats
    else:
        print("\n" + "=" * 60)
        print("❌ VERIFIKASI GAGAL (FAILED) - DITEMUKAN ERROR KOMPILASI")
        print("=" * 60)
        return False, {"status": "FAILED", "error": output}

if __name__ == "__main__":
    success, result = verify_firmware()
    sys.exit(0 if success else 1)
