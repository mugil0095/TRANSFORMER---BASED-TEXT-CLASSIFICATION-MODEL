import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained('model')
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def batch_inference(input_file, output_file):
    data = pd.read_csv(input_file)
    texts = data['review'].tolist()  # Use 'review' for text
    predictions = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(prediction)

    data['prediction'] = predictions
    data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    batch_inference('data/batch_input.csv', 'data/batch_output.csv')
