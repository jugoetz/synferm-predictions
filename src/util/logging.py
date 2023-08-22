import datetime
import random
from typing import Optional


def generate_run_id():
    """
    Return an ID consisting of the current datetime and a 6-digit random part.

    Returns:
        str: id in the format "YYYY-MM-DD-HHMMSS_r" where r is a 6 digit random integer (100000-999999).
    """
    curr_datetime = datetime.datetime.now()
    return f"{curr_datetime.date()}-{curr_datetime.hour:02d}{curr_datetime.minute:02d}{curr_datetime.second:02d}_{random.randint(100000, 999999)}"


def concatenate_to_dict_keys(
    dictionary: dict, prefix: Optional[str] = None, suffix: Optional[str] = None
) -> dict:
    """
    Concatenates keys of a dictionary with a prefix and/or suffix.
    """
    new = {}
    if prefix and suffix:
        for k, v in dictionary.items():
            new[prefix + k + suffix] = v
    elif prefix:
        for k, v in dictionary.items():
            new[prefix + k] = v
    elif suffix:
        for k, v in dictionary.items():
            new[k + suffix] = v
    else:
        new = dictionary
    return new
