from sklearn.datasets import load_iris
import pandas as pd

def load_data():
    iris = load_iris()
    X = iris.data  # 获取所有特征
    y = iris.target  # 获取目标类别
    feature_names = iris.feature_names  # 特征名称
    target_names = iris.target_names  # 类别名称

    # 只选择 Setosa (0) 和 Versicolor (1) 两类
    binary_mask = (y == 0) | (y == 1)  # Setosa(0) 和 Versicolor(1)
    X_binary = X[binary_mask, :]
    y_binary = y[binary_mask]

    # 创建DataFrame，方便后续分析
    df = pd.DataFrame(X_binary, columns=feature_names)
    df['species'] = y_binary
    df['species_name'] = [target_names[i] for i in y_binary]

    return X_binary, y_binary, feature_names, target_names, df
