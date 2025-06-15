import warnings
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score


def precision_recall_score(y_true, y_pred, ow_label, th):
    TP, FP, FN = 0, 0, 0

    for true_label, pred in zip(y_true, y_pred):
        # y_pred is the probability distribution for each category.
        max_prob = np.max(pred)
        pred_class = np.argmax(pred)

        if true_label != ow_label:
            if pred_class != ow_label and max_prob >= th:
                # True Example: Neither label nor prediction is ow_label
                # and the predicted probability exceeds the threshold.
                TP += 1
            else:
                # False negative example: label is not ow_label but
                # predicted as ow_label or predicted probability does not reach threshold.
                FN += 1
        elif true_label == ow_label:
            if pred_class != ow_label and max_prob >= th:
                # False positive example: label is ow_label but prediction
                # is not ow_label and prediction probability exceeds threshold.
                FP += 1

    # Prevent division by zero.
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
