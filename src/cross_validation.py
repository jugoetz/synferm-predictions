from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import KFold, ShuffleSplit
from torch.utils.data import DataLoader

from src.data.dataloader import collate_fn
from src.train import train, train_sklearn
from src.util.io import index_from_file
from src.util.logging import generate_run_id


def cross_validate(
    data,
    hparams,
    strategy="KFold",
    n_folds=0,
    train_size=0.9,
    split_files=None,
    return_fold_metrics=False,
    run_test=False,
    tags=None,
    job_type=None,
):
    """
    Trains a model under cross-validation. Returns the validation metrics' mean and std.

    Args:
        data (torch.utils.data.Dataset): Data set to run CV on.
        hparams (dict): Model configuration. See model for details.
        strategy (str): CV strategy. Supported: {"KFold", "predefined", "random"}. Defaults to "KFold".
        n_folds (int): Number of folds. Must be > 2 for strategy "KFold". Defaults to 0.
        train_size (float): Relative size of training set for random split. Used only for strategy "random", otherwise
            ignored. Defaults to 0.9.
        split_files (list): Splits to use. Expects a list of dictionaries, where each dictionary represents one fold
            e.g. [{"train": <path_to_train>, "val": <path_to_val>, "test": <path_to_test>}, ...]
            the dictionaries must contain keys "train" and "val" and can contain an arbitrary number of keys starting
            with "test"
        return_fold_metrics (bool, optional): Whether to return full train and val metrics for all folds.
            Defaults to False.
        run_test (bool, optional): Whether to run test set after training. Defaults to False.
        tags (list, optional): List of tags to add to the run in wandb. Defaults to None.
        job_type (str, optional): Type of job for wandb. Defaults to None.

    Returns:
        dict: Validation metrics, aggregated across folds (mean and standard deviation).
        dict: All metrics returned by train(), for all folds. Only returned if return_fold_metrics is True.
    """

    # generate run_id
    cv_run_id = generate_run_id()

    # set up splitter
    if strategy == "KFold":
        if n_folds < 2:
            raise ValueError("n_folds must be > 1 for cross-validation.")
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        # instead of splitting the data itself we only generate the split indices here
        # (for compatibility with predefined)
        split_idx = [
            {"train": fold[0], "val": fold[1]}
            for fold in splitter.split(list(range(len(data))))
        ]
    elif strategy == "predefined":
        # load indices from file
        split_idx = [
            {k: index_from_file(v) for k, v in fold.items()} for fold in split_files
        ]
    elif strategy == "random":
        # create a single random split with the given train_size
        splitter = ShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
        split_idx = [
            {"train": fold[0], "val": fold[1]}
            for fold in splitter.split(list(range(len(data))))
        ]
    else:
        raise ValueError(f"Invalid strategy '{strategy}'")

    # dict to hold fold metrics
    metrics = defaultdict(list)

    # iterate folds
    for i, fold in enumerate(split_idx):
        if strategy == "ShuffleSplit" and i >= n_folds:
            break  # exit loop for "endless" splitters like ShuffleSplit

        # generate run_id
        fold_run_id = f"{cv_run_id}_fold{i}"

        # instantiate DataLoaders
        data_splitted = {k: [data[i] for i in v] for k, v in fold.items()}
        train_dl = DataLoader(
            data_splitted["train"], batch_size=32, shuffle=True, collate_fn=collate_fn
        )
        val_dl = DataLoader(data_splitted["val"], batch_size=32, collate_fn=collate_fn)
        test_dls = (
            {
                k: DataLoader(v, batch_size=32, collate_fn=collate_fn)
                for k, v in data_splitted.items()
                if k.startswith("test")
            }
            if run_test
            else None
        )

        # train and validate model
        _, fold_metrics = train(
            train_dl,
            val_dl,
            hparams,
            test_dls=test_dls,
            run_id=fold_run_id,
            run_group=cv_run_id,
            return_metrics=True,
            tags=tags,
            job_type=job_type,
        )

        for k, v in fold_metrics.items():
            metrics[k].append(v)  # append fold metrics to list

    # aggregate fold metrics
    # stack tensors, but remove underscore metrics (timestamp etc.)
    metrics = {k: torch.stack(v) for k, v in metrics.items() if not k.startswith("_")}
    # statistics on all folds
    metrics_return = {k + "_mean": torch.mean(v) for k, v in metrics.items()}
    metrics_std = {k + "_std": torch.std(v) for k, v in metrics.items()}
    metrics_return.update(metrics_std)

    if return_fold_metrics:
        return metrics_return, metrics
    else:
        return metrics_return


def cross_validate_sklearn(
    data,
    hparams,
    strategy="KFold",
    n_folds=0,
    train_size=0.9,
    split_files=None,
    return_fold_metrics=False,
    run_test=False,
    tags=None,
    job_type=None,
    **kwargs,
):
    """
    Trains a sklearn model under cross-validation. Returns the validation metrics' mean/std and if test sets are given,
    test mean/std.

    The CV splits must be supplied as files containing indices (see arg split_files).

    Function will report evaluation metrics on all sets, i.e. val, and any sets starting with "test".
    Only val may be used to compare models before selection. To avoid computing test scores,
    do not pass test set indices.

    Args:
        data (torch.utils.data.Dataset): Data set to run CV on.
        hparams (dict): Model configuration. See model for details.
        strategy (str): CV strategy. Supported: {"KFold", "predefined", "random"}. Defaults to "KFold".
        n_folds (int): Number of folds. Must be > 2 for strategy "KFold". Defaults to 0.
        train_size (float): Relative size of training set for random split. Used only for strategy "random", otherwise
            ignored. Defaults to 0.9.
        split_files (list): Splits to use. Expects a list of dictionaries, where each dictionary represents one fold
            e.g. [{"train": <path_to_train>, "val": <path_to_val>, "test": <path_to_test>}, ...]
            the dictionaries must contain keys "train" and "val" and can contain an arbitrary number of keys starting
            with "test"
        return_fold_metrics (bool, optional): Whether to additionally return full train and val metrics for all folds.
            Defaults to False.
        run_test (bool, optional): Whether to run test set after training. Defaults to False.
        tags (list, optional): List of tags to add to the run in wandb. Defaults to None.
        job_type (str, optional): Type of job for wandb. Defaults to None.

    Returns:
        dict: Validation metrics, aggregated across folds (mean and standard deviation).
        dict: All metrics returned by train(), for all folds. Only returned if return_fold_metrics is True.
    """

    # generate run_id
    cv_run_id = generate_run_id()

    # set up splitter
    if strategy == "KFold":
        if n_folds < 2:
            raise ValueError("n_folds must be > 1 for cross-validation.")
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        # instead of splitting the data itself we only generate the split indices here
        # (for compatibility with predefined)
        split_idx = [
            {"train": fold[0], "val": fold[1]}
            for fold in splitter.split(list(range(len(data))))
        ]
    elif strategy == "predefined":
        # load indices from file
        split_idx = [
            {k: index_from_file(v) for k, v in fold.items()} for fold in split_files
        ]
    elif strategy == "random":
        # create a single random split with the given train_size
        splitter = ShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
        split_idx = [
            {"train": fold[0], "val": fold[1]}
            for fold in splitter.split(list(range(len(data))))
        ]
    else:
        raise ValueError(f"Invalid strategy '{strategy}'")

    # dict to hold fold metrics
    metrics = defaultdict(lambda: np.zeros((len(split_files),)))

    # iterate folds
    for i, fold in enumerate(split_idx):
        if strategy == "ShuffleSplit" and i >= n_folds:
            break  # exit loop for "endless" splitters like ShuffleSplit

        # generate run_id
        fold_run_id = f"{cv_run_id}_fold{i}"

        # instantiate Dataset splits instead of DataLoaders
        data_splitted = {k: [data[i] for i in v] for k, v in fold.items()}
        train_data = data_splitted["train"]
        val_data = data_splitted["val"]
        test_data = (
            {k: v for k, v in data_splitted.items() if k.startswith("test")}
            if run_test
            else None
        )

        # train and validate model
        _, fold_metrics = train_sklearn(
            train_data,
            val_data,
            hparams,
            test=test_data,
            run_id=fold_run_id,
            run_group=cv_run_id,
            return_metrics=True,
            tags=tags,
            job_type=job_type,
        )

        for k, v in fold_metrics.items():
            metrics[k][i] = v  # insert fold metrics into array

    # aggregate fold metrics (aggregation is not possible for all of them. E.g. makes no sense to aggregate roc curves)
    aggregated_metrics = {}
    for k, v in metrics.items():
        if k.endswith(
            ("precision", "recall", "f1", "accuracy", "roc_auc", "loss")
        ):  # these are the metrics that can be aggregated
            aggregated_metrics[k + "_mean"], aggregated_metrics[k + "_std"] = (
                v.mean(),
                v.std(),
            )

    if return_fold_metrics:
        return aggregated_metrics, metrics
    else:
        return aggregated_metrics
