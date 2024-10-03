
# Transformer-Based Text Classification Pipeline

## Overview
This Assignment implements a continuous training and deployment pipeline for a transformer-based text classification model. The pipeline supports both **live** and **batch** inference, and retrains the model upon detecting staleness or performance degradation using monitoring tools like **MLflow**.

## Prerequisites

1. **Python**: Ensure Python 3.8+ is installed on your system.
2. **Dependencies**: Install required Python libraries. Run the following command to install all dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure
```
textclassification
 ├── data
 │    ├── IMDB Dataset.csv         # Dataset for training and validation
 │    ├── batch_input.csv          # Input for batch inference
 │    └── batch_output.csv         # Output generated after batch inference
 ├── model
 │    ├── config.json              # Generated automatically during training
 │    └── model.safetensor         # Model weights generated automatically
 ├── scripts
 │    ├── __init__.py              # Initialization script
 │    ├── preprocess.py            # Data preprocessing script
 │    ├── train.py                 # Model training script
 │    ├── inference.py             # Batch inference script
 │    ├── live_inference.py        # FastAPI for live inference
 │    └── monitoring.py            # Monitoring script for performance degradation
 ├── pipeline.py                   # Master pipeline to orchestrate the execution of steps
 └── requirements.txt              # List of required Python packages
```

## Deployment Instructions

### 1. **Set Up Monitoring**

The system monitors the model performance and sends an email alert when the performance degrades. To set up monitoring:
1. **Edit the `monitoring.py` file** with your email configuration:
    ```python
    EMAIL_USER = "your_email@gmail.com"
    EMAIL_PASSWORD = "your_password"
    TO_EMAIL = "recipient_email@gmail.com"
    ```
2. Adjust the **accuracy threshold** in `monitoring.py` as needed:
    ```python
    ACCURACY_THRESHOLD = 0.75  # Retrain if accuracy drops below this
    ```

### 2. **Run the Pipeline**

The entire pipeline can be run using the `pipeline.py` script. This will orchestrate the following steps:
   - Preprocess the dataset.
   - Train the model.
   - Run batch inference.
   - Launch the live inference server.
   - Start monitoring for performance degradation.

To run the pipeline:

```bash
python pipeline.py
```

This will:
1. Start monitoring in the background.
2. Preprocess the data (`preprocess.py`).
3. Train the model (`train.py`).
4. Run batch inference to generate predictions (`inference.py`).
5. Start the FastAPI server for live inference (`live_inference.py`).

### 3. **Batch Inference**

To run batch inference on new data, place your input data in `data/batch_input.csv`. Ensure the input format matches the original dataset (e.g., a column named `review` for text).

Then, run:

```bash
python scripts/inference.py
```

The predictions will be saved to `data/batch_output.csv`.

### 4. **Live Inference**

To deploy the model for live inference, a FastAPI service is available:

1. Ensure the `pipeline.py` script has started the FastAPI service, or you can run it directly:
   ```bash
   uvicorn scripts.live_inference:app --reload
   ```

2. Access the API at `http://127.0.0.1:8000`. 
   
   To get a prediction, use the `/predict` endpoint `http://127.0.0.1:8000/docs`:
   ```bash
   POST /predict
   ```
   with a JSON body:
   ```json
   {
     "text": "Your text input here"
   }
   ```

3. The response will return the sentiment prediction (positive/negative).

### 5. **Retraining the Model**
   
The model will retrain automatically when performance degrades based on monitoring. You can manually initiate retraining by running:

```bash
python scripts/train.py
```

This will load the dataset, preprocess it, and retrain the model, saving the updated weights and configuration to the `model` directory.

### 6. **Monitor Model Performance**

The monitoring script (`monitoring.py`) checks the model's performance periodically and sends an email if the accuracy falls below the set threshold.

If the model's performance drops, retraining will be triggered automatically. 

You can run the monitoring script manually as well:

```bash
python scripts/monitoring.py
```

## Summary of Commands:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the full pipeline**:
   ```bash
   python pipeline.py
   ```

3. **Run batch inference**:
   ```bash
   python scripts/inference.py
   ```

4. **Run live inference**:
   ```bash
   uvicorn scripts.live_inference:app --reload
   ```

5. **Monitor model performance**:
   ```bash
   python scripts/monitoring.py
   ```

6. **Retrain the model**:
   ```bash
   python scripts/train.py
   ```

## Notes

- The `pipeline.py` automates the entire workflow. It first starts monitoring, preprocesses the data, trains the model, performs batch inference, and finally starts the live inference server.
- You can adjust hyperparameters, batch size, or learning rate directly in the scripts as needed.
- Ensure the SMTP configuration in `monitoring.py` is set properly to receive email alerts for model staleness.
