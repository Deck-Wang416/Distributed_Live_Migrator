1. Loaded the pre-trained BERT-Base model using the Hugging Face Transformers library.
2. Retrieved the IMDB dataset from Kaggle for a sentiment classification task.
3. Selected 12,500 samples from the dataset: First, split the dataset into 80% training (40,000) and 20% testing (10,000). Then, randomly sampled 25% of each split, resulting in 10,000 training samples and 2,500 testing samples.
4. Enabled GPU acceleration using the Metal backend on an M2 chip.
5. Trained a sentiment classification model based on BERT-Base, using 20% of the training samples for validation and achieving approximately 81.3% accuracy on the test samples.
6. Managed large files (datasets and models sized 50–100 MB) using Git LFS for version control.
7. Implemented save_checkpoint and load_checkpoint functions to save and restore model state, optimizer state, and training progress for checkpoint management.
8. Resume from the last saved epoch after an interruption: if interrupted after saving the checkpoint for epoch 2, the training will resume from epoch 3, continuing from the saved state rather than the exact interruption point.
