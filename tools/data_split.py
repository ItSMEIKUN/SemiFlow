import os
import sys
from sklearn.model_selection import train_test_split
sys.path.append(os.path.join(".."))


def split_data(X, y, train_size, test_size, unlabel_size):
    """
    Stratified sampling splits the dataset.
    """
    # Stratified Split Training Set and Remaining Data.
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, stratify=y, train_size=train_size,)

    # Split the test set and residual data hierarchically.
    X_test, X_remaining, y_test, y_remaining = train_test_split(
        X_remaining, y_remaining, stratify=y_remaining, train_size=test_size)

    # Split out unlabeled data.
    X_unlabel, X_remaining, y_unlabel, y_remaining = train_test_split(
        X_remaining, y_remaining, stratify=y_remaining, train_size=unlabel_size)

    return X_train, y_train, X_test, y_test, X_unlabel
