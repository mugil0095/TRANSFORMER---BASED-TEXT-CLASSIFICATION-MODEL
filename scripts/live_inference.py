from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize the FastAPI app
app = FastAPI()

# Load pre-trained model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained('model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Define the request body model
class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API! Use POST /predict to get predictions."}

@app.post("/predict")
def predict(request: TextRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    sentiment = "positive" if prediction == 1 else "negative"
    
    return {"prediction": sentiment}
