import numpy as np
import os
import tensorflow as tf
from main_forecast import build_proposed_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def quick_test():
    print("\n" + "="*40)
    print("   RUNNING QUICK INTEGRITY TEST")
    print("="*40)
    
    print("1. Generating synthetic seismic data...")
    X_dummy = np.random.rand(20, 10, 8).astype(np.float32)
    y_dummy = np.random.rand(20).astype(np.float32)
    
    print("2. Building Physics-Guided Bi-LSTM model...")
    try:
        model = build_proposed_model(input_shape=(10, 8))
        model.summary(print_fn=lambda x: None)
        print("    Model architecture built successfully.")
    except Exception as e:
        print(f"  Model build failed: {e}")
        return

    print("3. Testing training loop (1 Epoch)...")
    try:
        model.fit(X_dummy, y_dummy, epochs=1, batch_size=4, verbose=0)
        print("    Training step completed successfully.")
    except Exception as e:
        print(f"  Training failed: {e}")
        return

    print("4. Testing inference/prediction...")
    try:
        pred = model.predict(X_dummy[:2], verbose=0)
        if pred.shape == (2, 1):
            print("    Inference successful. Output shape is correct.")
        else:
            print(f"  Inference shape mismatch: {pred.shape}")
    except Exception as e:
        print(f"   Inference failed: {e}")
        return

    print("\n" + "="*40)
    print("   QUICK TEST PASSED SUCCESSFULLY")
    print("="*40 + "\n")

if __name__ == "__main__":
    quick_test()