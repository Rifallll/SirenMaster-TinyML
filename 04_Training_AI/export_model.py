"""
Siren Classifier Deployment and Export Script
Converts the trained Keras model (.h5) into:
1. Standard float32 TensorFlow Lite model (.tflite)
2. Optimized quantized dynamic range TensorFlow Lite model (.tflite)
3. C/C++ Header file (siren_model_data.h) containing the bytes array for ESP32/microcontroller deployment.
"""

import os
import tensorflow as tf

def convert_and_export(h5_model_path="siren_classifier_model.h5"):
    if not os.path.exists(h5_model_path):
        raise FileNotFoundError(f"Keras model not found at {h5_model_path}. Please train the model first.")
        
    print(f"[*] Loading Keras model: {h5_model_path}...")
    model = tf.keras.models.load_model(h5_model_path)
    
    # 1. Standard float32 conversion
    print("[*] Converting to standard TFLite (Float32)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    float_tflite_path = "siren_classifier_model.tflite"
    with open(float_tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"  Saved float32 TFLite model to '{float_tflite_path}' ({len(tflite_model)} bytes)")
    
    # 2. Quantized dynamic range conversion
    print("[*] Converting to quantized TFLite (Dynamic Range)...")
    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter_quant.convert()
    
    quant_tflite_path = "siren_model_quant.tflite"
    with open(quant_tflite_path, "wb") as f:
        f.write(tflite_quant_model)
    print(f"  Saved quantized TFLite model to '{quant_tflite_path}' ({len(tflite_quant_model)} bytes)")
    
    # 3. Generate C header file
    print("[*] Generating C++ header file for microcontroller deployment...")
    header_path = "siren_model_data.h"
    
    with open(header_path, "w") as f:
        f.write("/*\n")
        f.write(" * Siren Classifier - Model Byte Array Header\n")
        f.write(" * Generated automatically by export_model.py\n")
        f.write(f" * Original Model: {h5_model_path}\n")
        f.write(f" * Model Format: TensorFlow Lite (Dynamic Range Quantized)\n")
        f.write(f" * Size: {len(tflite_quant_model)} bytes\n")
        f.write(" */\n\n")
        f.write("#ifndef SIREN_MODEL_DATA_H\n")
        f.write("#define SIREN_MODEL_DATA_H\n\n")
        
        # Add alignment for safety on microcontroller hardware architectures
        f.write("// Align the array for TensorFlow Lite Micro memory alignment requirements\n")
        f.write("#ifdef __has_attribute\n")
        f.write("#define MODEL_ALIGN __attribute__((aligned(4)))\n")
        f.write("#else\n")
        f.write("#define MODEL_ALIGN\n")
        f.write("#endif\n\n")
        
        f.write(f"const unsigned int g_siren_model_data_len = {len(tflite_quant_model)};\n\n")
        f.write("const unsigned char g_siren_model_data[] MODEL_ALIGN = {\n")
        
        # Write array elements in hex format, 12 bytes per line
        bytes_per_line = 12
        for i, val in enumerate(tflite_quant_model):
            if i % bytes_per_line == 0:
                f.write("    ")
            f.write(f"0x{val:02x}")
            if i < len(tflite_quant_model) - 1:
                f.write(", ")
            if (i + 1) % bytes_per_line == 0 or i == len(tflite_quant_model) - 1:
                f.write("\n")
                
        f.write("};\n\n")
        f.write("#endif // SIREN_MODEL_DATA_H\n")
        
    print(f"  Saved C++ header file to '{header_path}'")
    
    # 4. Generate C header file for Scaler
    print("[*] Generating C++ header file for scaler parameters...")
    scaler_path = "siren_scaler.joblib"
    if os.path.exists(scaler_path):
        import joblib
        scaler = joblib.load(scaler_path)
        scaler_header_path = "siren_scaler_values.h"
        with open(scaler_header_path, "w") as f:
            f.write("/*\n")
            f.write(" * Siren Classifier - Scaler Mean and Std values\n")
            f.write(" * Generated automatically by export_model.py\n")
            f.write(f" * Feature Count: {len(scaler.mean_)}\n")
            f.write(" */\n\n")
            f.write("#ifndef SIREN_SCALER_VALUES_H\n")
            f.write("#define SIREN_SCALER_VALUES_H\n\n")
            
            f.write(f"const int g_scaler_features_len = {len(scaler.mean_)};\n\n")
            
            # Write Mean array
            f.write("const float g_scaler_mean[] = {\n")
            for i, val in enumerate(scaler.mean_):
                if i % 8 == 0:
                    f.write("    ")
                f.write(f"{val:.8f}f")
                if i < len(scaler.mean_) - 1:
                    f.write(", ")
                if (i + 1) % 8 == 0 or i == len(scaler.mean_) - 1:
                    f.write("\n")
            f.write("};\n\n")
            
            # Write Scale array (Standard Deviation)
            f.write("const float g_scaler_std[] = {\n")
            for i, val in enumerate(scaler.scale_):
                if i % 8 == 0:
                    f.write("    ")
                f.write(f"{val:.8f}f")
                if i < len(scaler.scale_) - 1:
                    f.write(", ")
                if (i + 1) % 8 == 0 or i == len(scaler.scale_) - 1:
                    f.write("\n")
            f.write("};\n\n")
            
            f.write("#endif // SIREN_SCALER_VALUES_H\n")
        print(f"  Saved C++ header file for scaler to '{scaler_header_path}'")
        
    print("\n[SUCCESS] All models and scaling parameters exported and deployment headers ready!")

if __name__ == "__main__":
    convert_and_export()
