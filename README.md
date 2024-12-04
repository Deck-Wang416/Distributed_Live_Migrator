# Distributed_Live_Migrator
A distributed deep learning training tool developed based on the FairScale framework, designed to enable real-time migration of BERT-Base training tasks across multiple nodes.

## Features
- **Real-time Migration**: Support for checkpoint saving and recovery during training.
- **Model Parallelism**: Utilize FairScale's pipeline parallelism to distribute training across GPUs/nodes.
- **Hugging Face Integration**: Train BERT-Base for binary classification tasks with Hugging Face Transformers.
- **Cloud Storage**: Azure Blob Storage and Azure Files support for distributed checkpoint management.

## Setup
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

2. **Prepare the dataset**:
   - Place the `IMDB_Dataset.csv` in the `data/` folder.
   - Ensure the dataset is correctly preprocessed with `scripts/preprocess.py`.

3. **Train the model**:
   ```bash
   python scripts/train.py

4. **Evaluate the model**:
   ```bash
   python scripts/evaluate.py

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
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Use Git LFS (Large File Storage)
This project uses Git LFS to manage large files like datasets and model weights efficiently.

**Install Git LFS**:
   ```bash
   git lfs install
