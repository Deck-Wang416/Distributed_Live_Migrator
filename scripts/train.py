import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from sklearn.model_selection import train_test_split
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from preprocess import preprocess_data
from save_and_load import save_model, save_checkpoint, load_checkpoint

def main():
    # Initialize distributed training
    dist.init_process_group(backend="gloo")  # Use Gloo backend for multi-CPU training
    local_rank = int(os.environ["LOCAL_RANK"])  # Local rank (used for device assignment)
    world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes (used for distributed training)
    device = torch.device("cpu")  # Since we're using CPUs, set device to "cpu"
    print(f"Using device {device}")

    # Load and preprocess the dataset
    train_texts, train_labels, test_texts, test_labels = preprocess_data(
        "data/IMDB_Dataset.csv", train_size=1000, test_size=500
    )

    # Split the training data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42
    )

    # Load tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Encode the data
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )

    # Create datasets
    train_dataset = TensorDataset(
        train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(train_labels)
    )
    val_dataset = TensorDataset(
        val_encodings["input_ids"], val_encodings["attention_mask"], torch.tensor(val_labels)
    )

    # Distributed sampler to ensure each process gets different data
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank)

    # DataLoader with DistributedSampler
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=8, sampler=val_sampler)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Load checkpoint if available
    checkpoint_dir = "checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_1_worker_{local_rank}.pt")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    
    # Move model to the selected device
    model.to(device)

    # Use DistributedDataParallel
    model = DDP(model, device_ids=[local_rank])

    # Training loop
    for epoch in range(start_epoch, 3):  # Start from the restored epoch
        model.train()
        total_loss = 0
        train_sampler.set_epoch(epoch)  # Ensure data is shuffled differently in each epoch
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

        # Save checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_checkpoint(model, optimizer, epoch + 1, file_path=f"{checkpoint_dir}/checkpoint_epoch_{epoch + 1}_worker_{local_rank}.pt")

    # Validation
    print("Starting validation...")
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

    # Save the final model
    save_model(model, tokenizer, "models/bert-base")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
