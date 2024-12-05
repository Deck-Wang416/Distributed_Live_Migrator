from transformers import BertForSequenceClassification, BertTokenizer
import os
import torch

def save_model(model, tokenizer, save_dir):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

def load_model(save_dir):
    model = BertForSequenceClassification.from_pretrained(save_dir)
    tokenizer = BertTokenizer.from_pretrained(save_dir)
    print(f"Model loaded from {save_dir}")
    return model, tokenizer

def save_checkpoint(model, optimizer, epoch, file_path):
    """
    保存模型、优化器状态和当前 epoch 到指定路径。
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
    从指定路径加载 checkpoint，恢复模型、优化器状态和当前 epoch。

    参数:
    - file_path (str): Checkpoint 文件路径。
    - model (torch.nn.Module): 要恢复的模型。
    - optimizer (torch.optim.Optimizer): 要恢复的优化器。

    返回:
    - int: 恢复的 epoch 值。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")
    
    checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    
    print(f"Checkpoint loaded from {file_path}")
    return epoch
