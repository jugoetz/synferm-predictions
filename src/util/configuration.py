import os

import yaml


def get_config(file: os.PathLike) -> dict:
    with open(file, "r") as f:
        conf = yaml.safe_load(f)
    return conf
