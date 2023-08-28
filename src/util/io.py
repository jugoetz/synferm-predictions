import yaml
import pathlib
import re
from collections import defaultdict


def index_from_file(path):
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


def get_hparam_bounds(path):
    """Read hyperparameter bounds from a yaml file"""
    with open(path, "r") as file:
        bounds = yaml.safe_load(file)
    return bounds


def walk_split_directory(directory: pathlib.Path):
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
