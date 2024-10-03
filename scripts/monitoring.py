
import smtplib
from email.mime.text import MIMEText
import mlflow
import time

# Email configuration (replace with your own credentials)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = "XYZ@gmail.com"
EMAIL_PASSWORD = "XYZ12345"
TO_EMAIL = "XZY@gmail.com"

# Monitoring configuration
ACCURACY_THRESHOLD = 0.75  # Retrain if accuracy drops below this
CHECK_INTERVAL = 3600  # Check every hour

def send_email_alert(message):
    msg = MIMEText(message)
    msg["Subject"] = "Model Performance Alert"
    msg["From"] = EMAIL_USER
    msg["To"] = TO_EMAIL

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USER, [TO_EMAIL], msg.as_string())

def check_model_performance():
    print("Checking model performance...")
    
    # Get the latest run from MLflow
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()

    for experiment in experiments:
        runs = client.search_runs(experiment.experiment_id)
        if runs:
            latest_run = runs[0]
            accuracy = latest_run.data.metrics.get('accuracy', None)
            
            if accuracy and accuracy < ACCURACY_THRESHOLD:
                message = f"Model accuracy has dropped to {accuracy}. Consider retraining the model."
                send_email_alert(message)
                print(message)
            else:
                print(f"Model accuracy is {accuracy}, which is above the threshold.")

if __name__ == "__main__":
    while True:
        check_model_performance()
        time.sleep(CHECK_INTERVAL)  # Wait for the next check
