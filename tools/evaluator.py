import warnings
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score


def precision_recall_score(y_true, y_pred, ow_label, th):
    TP, FP, FN = 0, 0, 0

    for true_label, pred in zip(y_true, y_pred):
        # 假设 y_pred 是每个类别的概率分布
        max_prob = np.max(pred)
        pred_class = np.argmax(pred)

        if true_label != ow_label:
            if pred_class != ow_label and max_prob >= th:
                TP += 1  # 真正例: 标签与预测都不是 ow_label 且预测概率超过阈值
            else:
                FN += 1  # 假负例: 标签不是 ow_label 但是预测为 ow_label 或者预测概率未达阈值
        elif true_label == ow_label:
            if pred_class != ow_label and max_prob >= th:
                FP += 1  # 假正例: 标签是 ow_label 但是预测不是 ow_label 且预测概率超过阈值

    # 防止除以零的情况
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, F1_score


def measurement(y_true, y_pred, ow_label, th):
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    results = {}
    if th == 0:
        y_pred = np.argmax(y_pred, axis=1)
        results["Accuracy"] = round(accuracy_score(y_true, y_pred) * 100, 2)
    else:
        precision, recall, F1_score = precision_recall_score(y_true, y_pred, ow_label, th)
        results["Precision"] = round(precision * 100, 2)
        results["Recall"] = round(recall * 100, 2)
        results["F1_score"] = round(F1_score * 100, 2)
    return results
