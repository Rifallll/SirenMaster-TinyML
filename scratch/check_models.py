import numpy as np
import tensorflow as tf
import os

def check_tflite_model(model_path):
    print(f"Checking: {model_path}")
    if not os.path.exists(model_path):
        print("File not found!")
        return
        
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Input details:")
    print(input_details)
    print("Output details:")
    print(output_details)
    print("-" * 50)

# Check all available TFLite models
check_tflite_model(r"C:\Users\ASUS\Videos\DATASET\sirenmaster_main\model.tflite")
check_tflite_model(r"C:\Users\ASUS\Videos\DATASET\siren_model_quant.tflite")
check_tflite_model(r"C:\Users\ASUS\Videos\DATASET\siren_classifier_model.tflite")
