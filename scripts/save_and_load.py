from transformers import BertForSequenceClassification, BertTokenizer
import os
import torch

def save_model(model, tokenizer, save_dir):
    """
    Save the model and tokenizer to the specified directory.

    Args:
    - model (transformers.PreTrainedModel): The model to save.
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer to save.
    - save_dir (str): Directory to save the model and tokenizer.

    Returns:
    - None
    """
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

def load_model(save_dir):
    """
    Load the model and tokenizer from the specified directory.

    Args:
    - save_dir (str): Directory to load the model and tokenizer from.

    Returns:
    - model (transformers.PreTrainedModel): The loaded model.
    - tokenizer (transformers.PreTrainedTokenizer): The loaded tokenizer.
    """
    model = BertForSequenceClassification.from_pretrained(save_dir)
    tokenizer = BertTokenizer.from_pretrained(save_dir)
    print(f"Model loaded from {save_dir}")
    return model, tokenizer

def save_checkpoint(model, optimizer, epoch, file_path):
    """
    Save the model state, optimizer state, and current epoch to the specified file.

    Args:
    - model (torch.nn.Module): The model whose state to save.
    - optimizer (torch.optim.Optimizer): The optimizer whose state to save.
    - epoch (int): The current epoch to save.
    - file_path (str): Path to save the checkpoint file.

    Returns:
    - None
    """
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

    Args:
    - file_path (str): Path to the checkpoint file.
    - model (torch.nn.Module): The model to restore.
    - optimizer (torch.optim.Optimizer): The optimizer to restore.

    Returns:
    - int: The epoch restored from the checkpoint.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")
    
    checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    
    print(f"Checkpoint loaded from {file_path}")
    return epoch
