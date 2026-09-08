import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

def main():
    CACHE_PATH = 'siren_40x63_melspec_cache_lokal.npz'
    print("Loading cache...")
    data = np.load(CACHE_PATH)
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Reload scaling params
    global_mean = np.mean(data['X_train'], axis=(0, 1))
    global_std = np.std(data['X_train'], axis=(0, 1))
    global_std[global_std < 1e-6] = 1.0
    
    # Scale test data
    X_test_scaled = (X_test - global_mean) / global_std
    X_test_scaled = X_test_scaled[..., np.newaxis].astype(np.float32)
    
    # Load model
    print("Loading model...")
    interpreter = tf.lite.Interpreter(model_path="siren_model_quant.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    scale, zp = input_details[0]['quantization']
    out_scale, out_zp = output_details[0]['quantization']
    
    y_pred = []
    print("Evaluating...")
    for i in range(len(X_test_scaled)):
        feat = X_test_scaled[i:i+1]
        feat_q = np.round(feat / scale + zp).astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], feat_q)
        interpreter.invoke()
        probs_q = interpreter.get_tensor(output_details[0]['index'])[0]
        probs = (probs_q.astype(np.float32) - out_zp) * out_scale
        
        y_pred.append(np.argmax(probs))
        
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['AMBULANCE', 'FIRETRUCK', 'POLICE', 'NORMAL']))

if __name__ == '__main__':
    main()
