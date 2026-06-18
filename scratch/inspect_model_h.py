import os

model_h = r"c:\Users\ASUS\Videos\DATASET\sirenmaster_main\model.h"
if os.path.exists(model_h):
    with open(model_h, 'r', encoding='utf-8', errors='ignore') as f:
        lines = []
        for i in range(100):
            line = f.readline()
            if not line:
                break
            lines.append(line.strip())
        print("=== FIRST 100 LINES OF MODEL.H ===")
        for idx, l in enumerate(lines):
            print(f"{idx+1}: {l}")
            
    # Search for variable names or signatures
    with open(model_h, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        print("\n=== SEARCH RESULTS IN MODEL.H ===")
        import re
        defines = re.findall(r'#define\s+\w+\s+\S+', content)
        print("Defines found:", defines)
        arrays = re.findall(r'const\s+\w+\s+\w+\[[^\]]*\]', content)
        print("Arrays found:", arrays[:20])
else:
    print("model.h not found at path")
