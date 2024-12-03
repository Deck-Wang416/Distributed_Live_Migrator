import pandas as pd

def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    texts = data["text"].tolist()
    labels = data["label"].tolist()
    return texts, labels
