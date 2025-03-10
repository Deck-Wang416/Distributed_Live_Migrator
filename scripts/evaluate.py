import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score
from scripts.preprocess import preprocess_data
from torch import distributed as dist
from torch.utils.data import DistributedSampler

# Load and process data
train_texts, train_labels, test_texts, test_labels = preprocess_data("data/IMDB_Dataset.csv")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("models/bert-base")
model = BertForSequenceClassification.from_pretrained("models/bert-base")

# Create datasets and dataloaders
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
test_dataset = TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], torch.tensor(test_labels))
test_sampler = DistributedSampler(test_dataset, shuffle=False)  # Ensures data is distributed across nodes
test_loader = DataLoader(test_dataset, batch_size=8, sampler=test_sampler)

# Evaluation
device = torch.device("cpu")  # Since we are using CPUs, we don't need CUDA
model.to(device)

# Initialize the process group
dist.init_process_group(backend="gloo")
local_rank = int(os.environ["LOCAL_RANK"])
model = model.to(device)

print("Starting evaluation...")
predictions, true_labels = [], []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f"Validation accuracy: {accuracy:.4f}")
