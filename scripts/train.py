import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from sklearn.model_selection import train_test_split
from scripts.preprocess import preprocess_data
from scripts.save_and_load import save_model

# 检查设备 (MPS or CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Metal (MPS) backend is available. Using MPS for training.")
else:
    device = torch.device("cpu")
    print("Metal (MPS) backend is not available. Using CPU for training.")

# 加载并缩小数据集
train_texts, train_labels, test_texts, test_labels = preprocess_data("data/IMDB_Dataset.csv", train_size=1000, test_size=500)

# 将训练数据划分为训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

# 加载分词器和模型
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 数据编码
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

train_dataset = TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(train_labels))
val_dataset = TensorDataset(val_encodings["input_ids"], val_encodings["attention_mask"], torch.tensor(val_labels))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# 优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 模型移动到设备
model.to(device)

# 创建模型保存路径
os.makedirs("models", exist_ok=True)

# 训练
print("开始训练...")
for epoch in range(3):  # 示例 3 个 epoch
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

# 验证
print("开始验证...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Validation Accuracy: {accuracy:.4f}")

# 保存模型
print("Saving model to models/bert-base...")
save_model(model, tokenizer, "models/bert-base")
print("Model saved successfully!")
