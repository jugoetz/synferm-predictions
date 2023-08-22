from copy import deepcopy

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def calculate_metrics(y_true, y_pred, pred_proba=False, detailed=False):
    """
    Calculate a bunch of metrics for classification problems.
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        pred_proba: Whether y_pred is a probability, or a label. If False, y_pred is assumed to be a label and
            log_loss is not calculated. Defaults to False.
        detailed: Whether to include ROC curve, PR curve, and confusion matrix. Defaults to False.

    Returns:
        dict: Dictionary of metrics

    """

    if pred_proba is True:
        y_prob = deepcopy(y_pred)
        y_pred = np.argmax(y_pred, axis=1)

    metrics = {
        k: v
        for k, v in zip(
            ["precision", "recall", "f1"],
            precision_recall_fscore_support(
                y_true, y_pred, average="binary", pos_label=1
            ),
        )
    }
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["AUROC"] = roc_auc_score(y_true, y_pred)
    metrics["loss"] = log_loss(y_true, y_prob) if pred_proba else None

    if detailed is True:
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        metrics["roc_curve"] = roc_curve(y_true, y_pred)
        metrics["pr_curve"] = precision_recall_curve(y_true, y_pred)
    return metrics
