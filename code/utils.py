import os
from pathlib import Path


module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder


def get_path(fname=None):
    if not fname:
        return str(Path(module_path).parents[0])
    return str(Path(module_path).parents[0]) + fname
