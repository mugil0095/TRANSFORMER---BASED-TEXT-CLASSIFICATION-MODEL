from transformers import AutoModelForSequenceClassification
from transformers import AdamW
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import mlflow
from preprocess import load_and_preprocess_data

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_model(train_encodings, val_encodings, train_labels, val_labels, epochs=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare datasets and loaders
    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Load BERT model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained('distilbert/distilbert-base-uncased-finetuned-sst-2-english', num_labels=2)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # Start MLflow experiment
    mlflow.start_run()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()

            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == inputs['labels']).sum().item()
            total += len(inputs['labels'])

        accuracy = correct / total
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.4f}")

        # Log accuracy to MLflow
        mlflow.log_metric("accuracy", accuracy)

        # Validate the model
        validate_model(model, val_loader, device)

    # End MLflow run
    mlflow.end_run()

    # Save the trained model
    model.save_pretrained('model')
    return model

def validate_model(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == inputs['labels']).sum().item()
            total += len(inputs['labels'])
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    train_encodings, val_encodings, train_labels, val_labels = load_and_preprocess_data()
    train_model(train_encodings, val_encodings, train_labels, val_labels, epochs=2)