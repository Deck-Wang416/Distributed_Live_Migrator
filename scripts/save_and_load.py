from transformers import BertForSequenceClassification, BertTokenizer
import os
import torch
from torch import distributed as dist

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from DistributedDataParallel models"""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove "module." prefix
        new_state_dict[new_key] = v
    return new_state_dict

def save_model(model, tokenizer, save_dir):
    """
    Save the model and tokenizer to the specified directory.
    Ensures only the main process performs the save operation.
    """
    if dist.get_rank() == 0:  # Only the main process saves the model
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")
    else:
        print("Skipping model save as this is not the main process.")

def load_model(save_dir):
    """
    Load the model and tokenizer from the specified directory.
    """
    model = BertForSequenceClassification.from_pretrained(save_dir)
    tokenizer = BertTokenizer.from_pretrained(save_dir)
    print(f"Model loaded from {save_dir}")
    return model, tokenizer

def save_checkpoint(model, optimizer, epoch, rank):
    """
    Save the model state, optimizer state, and current epoch to the specified file for each worker.
    """
    checkpoint_dir = "/app/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    file_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_worker_{rank}.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at {file_path} for Worker {rank}")

def load_checkpoint(model, optimizer, rank):
    """
    Load the checkpoint from the specified file and restore the model, optimizer states, and epoch.
    Each worker loads its own checkpoint.
    """
    checkpoint_dir = "/app/checkpoints"
    latest_epoch = 0
    checkpoint_path = None

    # Search for the latest checkpoint for this worker
    for epoch in range(3, 0, -1):  # Search from latest to earliest
        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_worker_{rank}.pt")
        if os.path.exists(path):
            checkpoint_path = path
            latest_epoch = epoch
            break

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(remove_module_prefix(checkpoint["model_state_dict"]))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Worker {rank} restored from {checkpoint_path}")
        return latest_epoch
    else:
        print(f"No checkpoint found for Worker {rank}, starting from scratch.")
        return 0
