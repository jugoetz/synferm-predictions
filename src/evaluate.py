from copy import deepcopy

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    log_loss,
    balanced_accuracy_score,
)


def calculate_metrics(y_true, y_prob, task, label_names=None):
    """
    Calculate a bunch of metrics for classification problems.
    Args:
        y_true: True labels, shape (n_samples,) or (n_samples, n_labels)
        y_prob: Predicted probabilities
        task: binary, multiclass, or multilabel
        label_names: List of target names for multilabel classification. If none are given, will use numbers from 0.

    Returns:
        dict: Dictionary of metrics
    """

    metrics = {}
    if task == "binary":
        y_true = np.array(y_true).astype(np.int32)[
            :, 0
        ]  # y_true is a list of tensors that we convert to a 1D np.array
        y_pred = (y_prob[:, 1] > 0.5).astype(np.int32)

        metrics["f1"] = f1_score(y_true, y_pred, average="binary")
        metrics["recall"] = recall_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred)
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        metrics["auroc"] = roc_auc_score(y_true, y_prob[:, 1])
        metrics["loss"] = log_loss(y_true, y_prob[:, 1])

    elif task == "multilabel":
        # sklearn's multilabel predictions arrive as a list of numpy arrays
        y_prob = np.stack([y[:, 1] for y in y_prob], axis=1)
        y_pred = (y_prob > 0.5).astype(np.int32)
        y_true = np.array(y_true).astype(
            np.int32
        )  # y_true is a list of tensors that we convert to a 2D np.array

        if label_names is None:
            label_names = [str(i) for i in range(y_pred.shape[1])]
        for i, s in zip(range(y_pred.shape[1]), label_names):
            metrics[f"f1_target_{s}"] = f1_score(
                y_true[:, i], y_pred[:, i], average="binary"
            )
            metrics[f"recall_target_{s}"] = recall_score(y_true[:, i], y_pred[:, i])
            metrics[f"precision_target_{s}"] = precision_score(
                y_true[:, i], y_pred[:, i]
            )
            metrics[f"accuracy_target_{s}"] = accuracy_score(y_true[:, i], y_pred[:, i])
            metrics[f"balanced_accuracy_target_{s}"] = balanced_accuracy_score(
                y_true[:, i], y_pred[:, i]
            )
            metrics[f"auroc_target_{s}"] = roc_auc_score(y_true[:, i], y_prob[:, i])
            metrics[f"loss_target_{s}"] = log_loss(y_true[:, i], y_prob[:, i])

        metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro")
        metrics["recall_macro"] = precision_score(y_true, y_pred, average="macro")
        metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro")
        metrics["accuracy_macro"] = np.mean(
            [v for k, v in metrics.items() if k.startswith("accuracy_target_")]
        )
        metrics["balanced_accuracy_macro"] = np.mean(
            [v for k, v in metrics.items() if k.startswith("balanced_accuracy_target_")]
        )
        metrics["auroc_macro"] = roc_auc_score(y_true, y_prob, average="macro")
        metrics["loss"] = np.mean(
            [v for k, v in metrics.items() if k.startswith("loss_target_")]
        )

    elif task == "multiclass":
        y_pred = np.argmax(y_prob, axis=1)
        raise NotImplementedError("implement this if it is needed")
    return metrics
