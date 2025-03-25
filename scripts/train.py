import os
import sys
import torch
import torch.distributed.rpc as rpc
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from transformers.utils import logging
from sklearn.model_selection import train_test_split
from torch import distributed as dist
from preprocess import preprocess_data
from save_and_load import save_model, save_checkpoint, load_checkpoint
from kubernetes import client, config
from torch import nn

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
    # Parse RANK
    pod_name = os.environ["POD_NAME"]
    rank = int(pod_name.split("-")[-1])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cpu")

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ.get("MASTER_PORT", "29500")
    start_daemon = rank == 0
    store = dist.TCPStore(
        host_name=master_addr,
        port=int(master_port),
        world_size=world_size,
        is_master=start_daemon,
        timeout=torch.distributed.constants.default_pg_timeout
    )

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=60,
    )
    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options,
        store=store
    )

    # Load and preprocess the dataset
    train_texts, train_labels, _, _ = preprocess_data(
        "data/IMDB_Dataset.csv", train_size=1000, test_size=500
    )

    # Split the training data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42
    )

    # Set logging to reduce memory usage
    logging.set_verbosity_error()

    # Load tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir="/app/hf_cache")
    full_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, cache_dir="/app/hf_cache")

    encoder_layers = list(full_model.bert.encoder.layer)
    if rank == 0:
        model = FrontBert(full_model.bert.embeddings, encoder_layers)
    else:
        model = BackBert(encoder_layers, full_model.bert.pooler, full_model.classifier)

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

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Load checkpoint if available
    checkpoint_dir = "/app/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch = load_checkpoint(full_model, optimizer, rank)

    # Move model to the selected device
    model.to(device)

    dist.barrier()
    for epoch in range(start_epoch, 3):  # Start from the restored epoch
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()

            if rank == 0:
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
                    print("Retrying with worker2 as fallback...")
                    fut = rpc.rpc_async(
                        to="worker2",
                        func=remote_forward,
                        args=(outputs.detach(), labels, attention_mask)
                    )
                    loss, grad_output = fut.wait()

                outputs.backward(grad_output)
                optimizer.step()
                total_loss += loss.item()
            else:
                print("Rank 1 is ready to receive RPC requests.")
                dist.barrier()
                rpc.shutdown()
                return

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

        save_checkpoint(full_model, optimizer, epoch + 1, rank)

        # Synchronous training
        dist.barrier()

    dist.barrier()  # Ensure all nodes finish training before validation

    # Validation
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

    # Save the final model
    if rank == 0:
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

    rpc.shutdown()
    sys.exit(0)

def remote_forward(hidden_states, labels, attention_mask):
    device = torch.device("cpu")
    if not hasattr(remote_forward, "model"):
        model_name = "bert-base-uncased"
        full_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, cache_dir="/app/hf_cache")
        encoder_layers = list(full_model.bert.encoder.layer)
        remote_forward.model = BackBert(
            encoder_layers,
            full_model.bert.pooler,
            full_model.classifier
        ).to(device)
    hidden_states.requires_grad = True
    output = remote_forward.model(hidden_states, attention_mask)
    loss = torch.nn.functional.cross_entropy(output, labels)
    loss.backward()
    return loss, hidden_states.grad

if __name__ == "__main__":
    main()
