1. Loaded the pre-trained BERT-Base model using the Hugging Face API.
2. Found the IMDB dataset on Kaggle to perform a sentiment classification task.
3. Selected 10,000 samples from the dataset: 80% for training and 20% for testing.
4. Enabled GPU acceleration using the Metal framework for the M2 chip.
5. Trained a sentiment classification model based on BERT-Base, achieving an accuracy of 83%.
6. Uploaded files (datasets and models sized 50–100MB) using Git LFS.
7. Implemented train.py/save_checkpoint to save the training state locally.
