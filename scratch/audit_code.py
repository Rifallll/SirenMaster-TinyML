import os
import re

def audit_cpp_files():
    main_dir = r"c:\Users\ASUS\Videos\DATASET\sirenmaster_main"
    ino_path = os.path.join(main_dir, "sirenmaster_main.ino")
    model_h_path = os.path.join(main_dir, "model.h")
    
    print("=== AUDITING C++ FILES ===")
    
    # Check if files exist
    if not os.path.exists(ino_path):
        print(f"[ERROR] {ino_path} does not exist!")
        return False
    if not os.path.exists(model_h_path):
        print(f"[ERROR] {model_h_path} does not exist!")
        return False
        
    # Read model.h
    print("[*] Reading model.h...")
    with open(model_h_path, "r", encoding="utf-8", errors="ignore") as f:
        model_h_content = f.read()
        
    # Read sirenmaster_main.ino
    print("[*] Reading sirenmaster_main.ino...")
    with open(ino_path, "r", encoding="utf-8", errors="ignore") as f:
        ino_content = f.read()
        
    errors = []
    
    # 1. Check for MEL_MEAN, MEL_STD, HAMMING_WINDOW, MEL_FILTERBANK in model.h
    required_symbols_model = ["MEL_MEAN", "MEL_STD", "HAMMING_WINDOW", "MEL_FILTERBANK", "siren_model_data", "siren_model_data_len"]
    for sym in required_symbols_model:
        if sym in model_h_content:
            print(f"  [OK] Symbol '{sym}' found in model.h")
        else:
            errors.append(f"Symbol '{sym}' is MISSING from model.h!")
            
    # 2. Check defines
    required_defines = [
        "SAMPLE_RATE_HZ", "N_FFT_SIZE", "N_HOP_LENGTH", 
        "N_MFCC_COEFF", "N_MEL_FILTERS", "N_TIME_FRAMES", 
        "N_FFT_BINS", "AUDIO_SAMPLES", "NUM_CLASSES"
    ]
    for df in required_defines:
        if f"#define {df}" in model_h_content:
            print(f"  [OK] Define '{df}' found in model.h")
        else:
            errors.append(f"Define '{df}' is MISSING from model.h!")
            
    # 3. Check bracket balance in ino_content
    # Let's count open/close braces, parentheses, brackets
    braces_open = ino_content.count("{")
    braces_close = ino_content.count("}")
    parens_open = ino_content.count("(")
    parens_close = ino_content.count(")")
    
    print(f"[*] Brace count: open={braces_open}, close={braces_close}")
    print(f"[*] Parenthesis count: open={parens_open}, close={parens_close}")
    
    if braces_open != braces_close:
        errors.append(f"Mismatched curly braces: {{ is {braces_open}, }} is {braces_close}")
    if parens_open != parens_close:
        errors.append(f"Mismatched parentheses: ( is {parens_open}, ) is {parens_close}")
        
    # 4. Check for nested functions in .ino content
    # Look for functions declared inside other functions. We can do a basic check.
    # In C++, finding function declarations inside blocks is not easily done with simple regex,
    # but we can look for common keywords or signs.
    # We can also check if smartDetect contains lcdShowDetection declaration.
    if "void lcdShowDetection" in ino_content:
        # Check where it is declared. It should be outside.
        # Let's make sure it's not inside smartDetect
        smart_detect_match = re.search(r'int smartDetect\s*\([^)]*\)\s*\{', ino_content)
        if smart_detect_match:
            smart_detect_start = smart_detect_match.start()
            # find end of smartDetect by matching braces
            depth = 0
            smart_detect_end = -1
            for i in range(smart_detect_start, len(ino_content)):
                if ino_content[i] == '{':
                    depth += 1
                elif ino_content[i] == '}':
                    depth -= 1
                    if depth == 0:
                        smart_detect_end = i
                        break
            if smart_detect_end != -1:
                smart_detect_body = ino_content[smart_detect_start:smart_detect_end]
                if "lcdShowDetection" in smart_detect_body and "void lcdShowDetection" in smart_detect_body:
                    errors.append("lcdShowDetection function declaration is nested inside smartDetect!")
                    
    # 5. Look for duplicate blocks or trailing corruptions
    # Sometimes trailing characters are written
    if ino_content.strip().endswith("}") or ino_content.strip().endswith("};") or ino_content.strip().endswith("#endif"):
        print("  [OK] sirenmaster_main.ino file ends cleanly")
    else:
        errors.append("sirenmaster_main.ino might have trailing garbage characters at the end!")

    # Print final status
    if errors:
        print("\n=== AUDIT FAILED ===")
        for err in errors:
            print(f"[ERROR] {err}")
        return False
    else:
        print("\n=== AUDIT PASSED ===")
        print("[SUCCESS] All files are syntactically balanced, macros defined, and required arrays present!")
        return True

if __name__ == "__main__":
    audit_cpp_files()
