import os
import sys
from sklearn.model_selection import train_test_split
sys.path.append(os.path.join(".."))


def split_data(X, y, train_size, test_size, unlabel_size):
    """分层抽样拆分数据集。"""
    # 分层拆分训练集和剩余数据
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, stratify=y, train_size=train_size,)

    # 分层拆分测试集和剩余数据
    X_test, X_remaining, y_test, y_remaining = train_test_split(
        X_remaining, y_remaining, stratify=y_remaining, train_size=test_size)

    # 拆分出未标记数据
    X_unlabel, X_remaining, y_unlabel, y_remaining = train_test_split(
        X_remaining, y_remaining, stratify=y_remaining, train_size=unlabel_size)

    return X_train, y_train, X_test, y_test, X_unlabel
