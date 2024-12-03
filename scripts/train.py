import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from sklearn.model_selection import train_test_split
from scripts.preprocess import preprocess_data
from scripts.save_and_load import save_model

# 加载数据
texts, labels = preprocess_data("data/dataset.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# 加载分词器和模型
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 数据处理
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

train_dataset = torch.utils.data.TensorDataset(train_encodings["input_ids"], torch.tensor(train_labels))
val_dataset = torch.utils.data.TensorDataset(val_encodings["input_ids"], torch.tensor(val_labels))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(3):  # 示例 3 轮
    for batch in train_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# 保存模型
save_model(model, tokenizer, "models/bert-base")
