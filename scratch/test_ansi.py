import os
import sys
import time

# Enable ANSI
os.system('')
# Clear screen once at start
os.system('cls' if os.name == 'nt' else 'clear')

print("Flicker-free count test:")
print("------------------------")
for i in range(10):
    # Move cursor to home (top-left)
    sys.stdout.write("\033[H")
    sys.stdout.flush()
    # Print content
    print("Flicker-free count test:")
    print("------------------------")
    print(f"Current count: {i}/9")
    print("This should not flicker at all.")
    time.sleep(0.3)

print("\nSuccess!")
