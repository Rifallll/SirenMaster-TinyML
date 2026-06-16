import msvcrt
import sys
import time

print("=" * 60)
print("  DIAGNOSTIK INPUT KEYBOARD")
print("=" * 60)
print("Silakan tekan tombol apa saja di keyboard Anda (misalnya Y, T, 1, 2).")
print("Tekan tombol 'ESC' untuk keluar.")
print("-" * 60)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

while True:
    if msvcrt.kbhit():
        char = msvcrt.getch()
        print(f"Raw bytes: {char}")
        try:
            decoded = char.decode('utf-8')
            print(f"Decoded string: '{decoded}'")
            print(f"Decoded lower : '{decoded.lower()}'")
        except Exception as e:
            print(f"Failed to decode: {e}")
        print("-" * 60)
        
        # ESC key has ASCII value 27
        if char == b'\x1b':
            print("ESC ditekan. Keluar...")
            break
    time.sleep(0.05)
