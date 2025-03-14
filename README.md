# Distributed_Live_Migrator
A distributed deep learning training tool developed based on the FairScale framework, designed to enable real-time migration of BERT-Base training tasks across multiple nodes.

## Features
- **Real-time Migration**: Support for checkpoint saving and recovery during training.
- **Model Parallelism**: Utilize FairScale's pipeline parallelism to distribute training across GPUs/nodes.
- **Hugging Face Integration**: Train BERT-Base for binary classification tasks with Hugging Face Transformers.
- **Cloud Storage**: Azure Blob Storage and Azure Files support for distributed checkpoint management.
- **Kubernetes Deployment**: Fully integrated with Kubernetes, supporting distributed training in AKS.

## Setup
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

2. **Build and push Docker image**:
   ```bash
   docker buildx build --platform linux/amd64 -t deckwang/distributed-trainer:latest .
   docker push deckwang/distributed-trainer:latest

3. **Deploy to Kubernetes**:
   ```bash
   kubectl apply -f k8s/pvc-uploader.yaml
   kubectl apply -f k8s/persistent-volume.yaml
   kubectl apply -f k8s/checkpoint-pvc.yaml
   kubectl apply -f k8s/model-pvc.yaml
   kubectl apply -f k8s/headless-service.yaml
   kubectl apply -f k8s/statefulset.yaml
   kubectl apply -f k8s/debug-pod.yaml

4. **Monitor the deployment**:
   ```bash
   kubectl get pods
   kubectl logs -f distributed-trainer-0 -c trainer
   kubectl logs -f distributed-trainer-1 -c trainer

## Verify
1. **Check checkpoints**:
   ```bash
   kubectl exec -it distributed-trainer-0 -- /bin/bash
   ls -lh /app/checkpoints

2. **Check model**:
   ```bash
   kubectl exec -it debug-pod -- /bin/bash
   ls -lh /app/models

## Project Structure

```plaintext
Distributed_Live_Migrator/
├── data/                   # Dataset folder (e.g., IMDB_Dataset.csv)
├── models/                 # Saved model files (e.g., trained BERT checkpoints)
├── scripts/                # Training, preprocessing, and evaluation scripts
│   ├── preprocess.py       # Data preprocessing
│   ├── train.py            # Training the BERT model
│   ├── evaluate.py         # Model evaluation
│   └── save_and_load.py    # Checkpoint saving and loading
├── k8s/                    # Kubernetes deployment configuration
│   ├── headless-service.yaml   # Service configuration for inter-pod communication
│   ├── persistent-volume.yaml  # Persistent volume and claim for data storage
│   ├── pvc-uploader.yaml       # Utility for uploading dataset to PVC
│   ├── checkpoint-pvc.yaml      # PVC for checkpoint storage
│   ├── model-pvc.yaml           # PVC for model storage
│   ├── statefulset.yaml        # StatefulSet managing distributed training pods
│   ├── debug-pod.yaml          # Debug pod for verifying saved models
├── .dockerignore            # Ignore unnecessary files in Docker builds
├── .gitignore               # Ignore unnecessary files in Git repository
├── .gitattributes           # Configuration for Git LFS
├── Dockerfile               # Docker image definition for training
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── TODO.md                  # Project todo list
```
