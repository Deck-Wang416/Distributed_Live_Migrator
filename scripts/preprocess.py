import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(filepath, train_size=10000, test_size=2500):
    """
    预处理 IMDB 数据集：
    1. 划分数据集为训练集和测试集（80%/20%）。
    2. 随机采样指定数量的训练集和测试集。
    3. 转换标签为数值格式。

    参数：
    - filepath (str): 数据集的路径。
    - train_size (int): 训练集采样的数量。
    - test_size (int): 测试集采样的数量。

    返回：
    - train_texts (list): 训练文本。
    - train_labels (list): 训练标签。
    - test_texts (list): 测试文本。
    - test_labels (list): 测试标签。
    """
    # 读取数据
    data = pd.read_csv(filepath)

    # 划分为训练集和测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # 从训练集中随机采样指定数量，从测试集中随机采样指定数量
    train_data = train_data.sample(n=train_size, random_state=42)
    test_data = test_data.sample(n=test_size, random_state=42)

    # 将标签映射为数值（positive -> 1, negative -> 0）
    train_data['label'] = train_data['sentiment'].map({'positive': 1, 'negative': 0})
    test_data['label'] = test_data['sentiment'].map({'positive': 1, 'negative': 0})

    # 提取文本和标签
    train_texts = train_data['review'].tolist()
    train_labels = train_data['label'].tolist()
    test_texts = test_data['review'].tolist()
    test_labels = test_data['label'].tolist()

    return train_texts, train_labels, test_texts, test_labels
