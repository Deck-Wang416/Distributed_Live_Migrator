from transformers import BertForSequenceClassification, BertTokenizer

def save_model(model, tokenizer, save_dir):
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

def load_model(save_dir):
    model = BertForSequenceClassification.from_pretrained(save_dir)
    tokenizer = BertTokenizer.from_pretrained(save_dir)
    print(f"Model loaded from {save_dir}")
    return model, tokenizer
