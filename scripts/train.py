import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data
from save_and_load import save_model, save_checkpoint, load_checkpoint

def main():
    # Check device availability (MPS or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Metal (MPS) backend is available. Using MPS for training.")
    else:
        device = torch.device("cpu")
        print("Metal (MPS) backend is not available. Using CPU for training.")

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

    # Create datasets and data loaders
    train_dataset = TensorDataset(
        train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(train_labels)
    )
    val_dataset = TensorDataset(
        val_encodings["input_ids"], val_encodings["attention_mask"], torch.tensor(val_labels)
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Move model to the selected device
    model.to(device)

    # Simulate data parallelism (even with a single GPU)
    if torch.cuda.device_count() > 1:  # If you have multiple GPUs
        model = torch.nn.DataParallel(model)

    # Create necessary directories for saving checkpoints
    os.makedirs("checkpoints", exist_ok=True)

    # Check for existing checkpoint
    checkpoint_dir = "checkpoints"
    latest_checkpoint = None

    if os.path.exists(checkpoint_dir):
        # Find all .pt files in the directory
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        if checkpoints:
            # Identify the latest checkpoint
            latest_checkpoint = sorted(checkpoints)[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
            print(f"Resuming training from epoch {start_epoch + 1}")
        else:
            start_epoch = 0
    else:
        start_epoch = 0

    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, 3):  # Example: 3 epochs
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

        # Save checkpoint at the end of each epoch
        save_checkpoint(model, optimizer, epoch + 1, file_path=f"checkpoints/checkpoint_epoch_{epoch + 1}.pt")

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
    print("Saving model to models/bert-base...")
    save_model(model, tokenizer, "models/bert-base")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
