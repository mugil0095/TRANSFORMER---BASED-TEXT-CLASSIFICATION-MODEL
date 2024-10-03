import subprocess
import time

def run_monitoring():
    # Start monitoring in the background
    print("Starting monitoring...")
    subprocess.Popen(["python", "scripts/monitoring.py"])

def run_preprocessing():
    print("Running preprocessing...")
    subprocess.run(["python", "scripts/preprocess.py"])

def run_training():
    print("Running model training...")
    subprocess.run(["python", "scripts/train.py"])

def run_batch_inference():
    print("Running batch inference...")
    subprocess.run(["python", "scripts/inference.py"])

def run_live_inference():
    print("Starting live inference API...")
    subprocess.run(["uvicorn", "scripts.live_inference:app"])

if __name__ == "__main__":
    # Step 1: Start monitoring
    run_monitoring()

    # Step 2: Run data preprocessing
    run_preprocessing()

    # Step 3: Run model training
    run_training()

    # Step 4: Run batch inference
    run_batch_inference()

    # Step 5: Start the live inference API
    run_live_inference()
