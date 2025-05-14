# SYSTEM IMPORTS
from collections.abc import Sequence
from typing import Tuple
import os
import sys


# PYTHON PROJECT IMPORTS


def load_mono(file_path: str) -> Sequence[Sequence[str]]:
    data: Sequence[Sequence[str]] = list()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(line.strip().rstrip())
    return data


def load_parallel(path1: str, path2: str) -> Tuple[Sequence[Sequence[str]], Sequence[Sequence[str]]]:
    return load_mono(path1), load_mono(path2)

