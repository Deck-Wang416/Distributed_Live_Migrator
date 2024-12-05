import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score
from scripts.preprocess import preprocess_data

# 加载并处理数据
train_texts, train_labels, test_texts, test_labels = preprocess_data("data/IMDB_Dataset.csv")

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained("models/bert-base")
model = BertForSequenceClassification.from_pretrained("models/bert-base")

# 数据处理
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
test_dataset = TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], torch.tensor(test_labels))
test_loader = DataLoader(test_dataset, batch_size=8)

# 评估
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

print("开始验证...")
predictions, true_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f"验证集准确率: {accuracy:.4f}")
