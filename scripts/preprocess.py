import pandas as pd

def preprocess_data(filepath):
    # 读取数据
    data = pd.read_csv(filepath)
    
    # 转换标签为 0 和 1
    data['label'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    
    # 提取文本和标签
    texts = data['review'].tolist()
    labels = data['label'].tolist()
    
    return texts, labels
