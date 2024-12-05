import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(filepath, train_size=10000, test_size=2500):
    """
    Preprocess the IMDB dataset:
    1. Split the dataset into training and testing sets (80%/20%).
    2. Randomly sample the specified number of training and testing examples.
    3. Convert labels into numerical format.

    Args:
    - filepath (str): Path to the dataset file.
    - train_size (int): Number of samples to include in the training set.
    - test_size (int): Number of samples to include in the testing set.

    Returns:
    - train_texts (list): List of training texts.
    - train_labels (list): List of training labels.
    - test_texts (list): List of testing texts.
    - test_labels (list): List of testing labels.
    """
    # Load data
    data = pd.read_csv(filepath)

    # Split into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Randomly sample specified numbers of training and testing data
    train_data = train_data.sample(n=train_size, random_state=42)
    test_data = test_data.sample(n=test_size, random_state=42)

    # Map labels to numerical values (positive -> 1, negative -> 0)
    train_data['label'] = train_data['sentiment'].map({'positive': 1, 'negative': 0})
    test_data['label'] = test_data['sentiment'].map({'positive': 1, 'negative': 0})

    # Extract texts and labels
    train_texts = train_data['review'].tolist()
    train_labels = train_data['label'].tolist()
    test_texts = test_data['review'].tolist()
    test_labels = test_data['label'].tolist()

    return train_texts, train_labels, test_texts, test_labels
