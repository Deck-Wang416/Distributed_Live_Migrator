import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score
from scripts.preprocess import preprocess_data

# 加载数据
texts, labels = preprocess_data("data/IMDB_Dataset.csv")

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained("models/bert-base")
model = BertForSequenceClassification.from_pretrained("models/bert-base")

# 数据处理
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
dataset = torch.utils.data.TensorDataset(encodings["input_ids"], torch.tensor(labels))
data_loader = DataLoader(dataset, batch_size=8)

# 评估
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

predictions, true_labels = [], []
with torch.no_grad():
    for batch in data_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        outputs = model(input_ids=inputs)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
        true_labels.extend(labels.numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy}")
