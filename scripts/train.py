import os
import sys
import torch
import torch.distributed.rpc as rpc
from datetime import time, timedelta
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from transformers.utils import logging
from sklearn.model_selection import train_test_split
from torch import distributed as dist
from preprocess import preprocess_data
from save_and_load import save_model, save_checkpoint, load_checkpoint
from kubernetes import client, config
from torch import nn

# Processes input embeddings and the first six Transformer layers
class FrontBert(nn.Module):
    def __init__(self, embeddings, encoder_layers):
        super().__init__()
        self.embeddings = embeddings
        self.encoder = nn.ModuleList(encoder_layers[:6])

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        for layer_module in self.encoder:
            x = layer_module(x, attention_mask=extended_attention_mask)[0]
        return x

# Processes the remaining Transformer layers, pooling, and classification.
class BackBert(nn.Module):
    def __init__(self, encoder_layers, pooler, classifier):
        super().__init__()
        self.encoder = nn.ModuleList(encoder_layers[6:])
        self.pooler = pooler
        self.classifier = classifier

    def forward(self, hidden_states, attention_mask):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        for layer_module in self.encoder:
            hidden_states = layer_module(hidden_states, attention_mask=extended_attention_mask)[0]
        pooled_output = self.pooler(hidden_states)
        logits = self.classifier(pooled_output)
        return logits

def main():
    device = torch.device("cpu")

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        pod_name = os.environ["POD_NAME"]
        rank = int(pod_name.split("-")[-1])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        time.sleep(30)
    # Using the gloo backend.
    dist.init_process_group(backend='gloo', timeout=timedelta(seconds=120))
    
    # Initialize the RPC framework with TensorPipe as underlying protocol.
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=60
    )
    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )

    # Data preprocessing and loading.
    train_texts, train_labels, _, _ = preprocess_data(
        "data/IMDB_Dataset.csv", train_size=1000, test_size=500
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42
    )
    logging.set_verbosity_error()

    # Load tokenizer and pre-trained BERT model.
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir="/app/hf_cache")
    full_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, cache_dir="/app/hf_cache")
    encoder_layers = list(full_model.bert.encoder.layer)

    # Split model
    if rank == 0:
        model = FrontBert(full_model.bert.embeddings, encoder_layers)
    else:
        model = BackBert(encoder_layers, full_model.bert.pooler, full_model.classifier)

    # Encode data and create DataLoaders.
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    train_dataset = TensorDataset(
        train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(train_labels)
    )
    val_dataset = TensorDataset(
        val_encodings["input_ids"], val_encodings["attention_mask"], torch.tensor(val_labels)
    )
    train_loader = DataLoader(train_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)

    # Initialize optimizer and load checkpoint if available.
    optimizer = AdamW(model.parameters(), lr=5e-5)
    checkpoint_dir = "/app/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch = load_checkpoint(full_model, optimizer, rank)

    model.to(device)
    if rank != 0:
        global worker_model
        worker_model = model

    dist.barrier()

    # Perform forward on FrontBert and use RPC for BackBert computation.
    if rank == 0:
        for epoch in range(start_epoch, 3):
            total_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                try:
                    fut = rpc.rpc_async(
                        to="worker1",
                        func=remote_forward,
                        args=(outputs.detach(), labels, attention_mask)
                    )
                    loss, grad_output = fut.wait()
                except Exception as e:
                    print(f"[Warning] RPC to worker1 failed: {e}")
                    rpc.shutdown()
                    sys.exit(1)
                outputs.backward(grad_output)
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
            save_checkpoint(full_model, optimizer, epoch + 1, rank)
            dist.barrier()
        
        save_model(full_model, tokenizer, "models/bert-base")
        print("Model saved successfully!")

        def delete_statefulset():
            config.load_incluster_config()
            api_instance = client.AppsV1Api()
            namespace = "default"
            name = "distributed-trainer"
            try:
                api_instance.delete_namespaced_stateful_set(name, namespace)
                print(f"StatefulSet {name} deleted successfully.")
            except Exception as e:
                print(f"Failed to delete StatefulSet: {e}")

        print("Training complete. Deleting StatefulSet.")
        delete_statefulset()
    else:
        # Only participate in barrier synchronization and serve RPC requests.
        for epoch in range(start_epoch, 3):
            dist.barrier()
        if rank == 1:
            print("Starting validation.")
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = [b.to(device) for b in batch]
                    outputs = model(input_ids, attention_mask)
                    predictions = torch.argmax(outputs, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
            accuracy = correct / total
            print(f"Validation Accuracy: {accuracy:.4f}")

    dist.barrier()
    rpc.shutdown()
    sys.exit(0)

# Executed on worker nodes for back-end computation.
def remote_forward(hidden_states, labels, attention_mask):
    global worker_model
    # Lazy Loading
    if not hasattr(remote_forward, "model"):
        remote_forward.model = worker_model
    hidden_states.requires_grad = True
    output = remote_forward.model(hidden_states, attention_mask)
    loss = torch.nn.functional.cross_entropy(output, labels)
    loss.backward()
    return loss, hidden_states.grad

if __name__ == "__main__":
    main()
