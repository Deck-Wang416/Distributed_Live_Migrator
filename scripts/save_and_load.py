from transformers import BertForSequenceClassification, BertTokenizer
import os
import torch
from torch import distributed as dist

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

def save_checkpoint(model, optimizer, epoch, file_path):
    """
    Save the model state, optimizer state, and current epoch to the specified file.
    """
    if dist.get_rank() == 0:  # Only the main process saves the checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved at {file_path}")

def load_checkpoint(file_path, model, optimizer):
    """
    Load the checkpoint from the specified file and restore the model, optimizer states, and epoch.
    """
    if os.path.exists(file_path):
        checkpoint = torch.load(file_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded from {file_path}")
        return epoch
    else:
        print(f"Checkpoint file not found: {file_path}")
        return 0
