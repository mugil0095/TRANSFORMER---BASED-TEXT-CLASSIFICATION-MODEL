import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def load_and_preprocess_data(filepath='data\IMDB Dataset.csv', test_size=0.2):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    texts = df['review'].tolist()  # Use 'review' for text
    labels = df['sentiment'].tolist()  # Use 'sentiment' for labels

    # Convert labels to numerical format if necessary
    label_mapping = {'positive': 1, 'negative': 0}
    labels = [label_mapping[label] for label in labels]

    # Split dataset into training and validation sets
    print("Splitting data into training and validation sets...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased-finetuned-sst-2-english')

    # Tokenize and encode text data
    print("Tokenizing and encoding text data...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    print("Data preprocessing completed.")
    return train_encodings, val_encodings, train_labels, val_labels


if __name__ == "__main__":
    load_and_preprocess_data()
