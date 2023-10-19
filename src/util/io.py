import os
import pathlib
import re
from collections import defaultdict
from typing import Iterable, Union, Tuple, List

import numpy as np
import pandas as pd
import torch
import yaml

from src.util.definitions import PRED_DIR, LOG_DIR


def index_from_file(path: Union[str, os.PathLike]) -> List[int]:
    """
    Load a list of newline-separated integer indices from a file.

    If the file has a single-line header it will be ignored.
    If the fine has a multi-line header, a ValueError is raised.
    """
    try:
        with open(path, "r") as file:
            indices = [int(l.strip("\n")) for l in file.readlines()]
    except ValueError:
        with open(path, "r") as file:
            next(file)  # skip the first line
            indices = [int(l.strip("\n")) for l in file.readlines()]
    return indices


def get_hparam_bounds(path: Union[str, os.PathLike]) -> dict:
    """Read hyperparameter bounds from a yaml file"""
    with open(path, "r") as file:
        bounds = yaml.safe_load(file)
    return bounds


def walk_split_directory(directory: pathlib.Path) -> List[dict]:
    """
    Walk a directory containing csv files to identify all split files therein.
    We expect the naming convention `fold<fold_nr>_<train/val/test>.csv`.

    Args:
        directory (pathlib.Path): Directory to walk.

    Returns:
        list: List of dictionaries where the i-th list element represents the i-th fold.
    """
    split_indices = defaultdict(dict)
    name_regex = re.compile(r"fold(\d+)_(\w*).csv")

    for file in directory.iterdir():
        try:
            fold_idx, name = re.search(name_regex, file.name).groups()
            if (
                "train" in name or "test" in name or "val" in name
            ):  # ignore any other matches
                split_indices[int(fold_idx)][name] = file
        except (
            AttributeError
        ):  # raised when regex doesn't match. Just ignore these files
            pass
    # instead of the defaultdict, we return a list. We sort by dict key to ensure the list is in fold order.
    return [i[1] for i in sorted(split_indices.items())]


def save_predictions(
    run_id: str,
    indices: Iterable[Iterable[int]],
    preds: Union[List[torch.Tensor], Tuple[torch.Tensor]],
    dataloader_idx: str,
):
    """
    Save predictions to disk

    Args:
        run_id (str): Unique identifier of the run
        indices (Iterable): Indices for the predictions, e.g. [[1],[2],[3|,[4],...] or corresponding long-format vector.
        preds (list or tuple): List or tuple of tensors
        dataloader_idx (str): Index of the Dataloader, e.g. "train", "val", "test"
    """
    # save the predictions
    filepath = PRED_DIR / run_id / f"{dataloader_idx}_preds_last.csv"
    filepath.parent.mkdir(exist_ok=True, parents=True)
    ind = np.array([i for ind in indices for i in ind]).reshape(
        (-1, 1)
    )  # shape (n_samples, 1)
    preds = torch.concatenate(preds).cpu().numpy()  # shape (n_samples, n_labels)
    columns = [
        "idx",
    ] + [f"pred_{n}" for n in range(preds.shape[1])]
    pd.DataFrame(np.hstack((ind, preds)), columns=columns).astype(
        {"idx": "int32"}
    ).to_csv(filepath, index=False)


def read_predictions(run_id: str, dataloader_idx: str) -> pd.DataFrame:
    """
    Read predictions from disk

    Args:
        run_id (str): Unique identifier of the run
        dataloader_idx (str): Index of the Dataloader, e.g. "train", "val", "test"

    Returns:
        pd.DataFrame: Dataframe one column per label and original data set indices
    """
    filepath = PRED_DIR / run_id / f"{dataloader_idx}_preds_last.csv"
    return pd.read_csv(filepath, index_col="idx")


def save_best_hparams(hparams: dict, experiment_id: str) -> None:
    """
    Save best hyperparameters to disk

    Args:
        hparams (dict): Best hyperparameters
        experiment_id (str): Unique identifier that will be used as part of the filename
    """
    filepath = LOG_DIR / "hyperparameters" / f"{experiment_id}.csv"
    df = pd.json_normalize(hparams)
    df.to_csv(filepath, index=False)
