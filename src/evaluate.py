import numpy as np
import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    average_precision_score,
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
        metrics["loss"] = log_loss(y_true, y_prob[:, 1])
        metrics[f"avgPrecision"] = average_precision_score(y_true, y_prob[:, 1])

    elif task == "multilabel":
        if isinstance(y_prob, list):
            # sklearn's multilabel predictions arrive as a list of numpy arrays
            # we cut the (redundant) prediction of the negative class
            y_prob = np.stack([y[:, 1] for y in y_prob], axis=1)
        elif isinstance(y_prob, torch.Tensor):
            # if we receive a torch.Tensor, convert it to numpy array
            y_prob = y_prob.numpy()
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
            metrics[f"loss_target_{s}"] = log_loss(y_true[:, i], y_prob[:, i])
            metrics[f"avgPrecision_target_{s}"] = average_precision_score(
                y_true[:, i], y_prob[:, i], average=None
            )
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
        metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro")
        metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro")
        metrics["accuracy_macro"] = np.mean(
            [v for k, v in metrics.items() if k.startswith("accuracy_target_")]
        )
        metrics["balanced_accuracy_macro"] = np.mean(
            [v for k, v in metrics.items() if k.startswith("balanced_accuracy_target_")]
        )
        metrics[f"avgPrecision_macro"] = average_precision_score(
            y_true, y_prob, average="macro"
        )

        metrics["loss"] = np.mean(
            [v for k, v in metrics.items() if k.startswith("loss_target_")]
        )

    elif task == "multiclass":
        y_pred = np.argmax(y_prob, axis=1)
        raise NotImplementedError("implement this if it is needed")
    return metrics
