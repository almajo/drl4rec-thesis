import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def save_obj(obj, name):
    """Saves given object as a pickle file"""
    if name[-4:] != ".pkl":
        name += ".pkl"
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """Loads a pickle file object"""
    with open(name, 'rb') as f:
        return pickle.load(f)


def load_embedding(path):
    return np.load(path)


def dir_is_empty(path):
    p = Path(path)
    return p.is_dir() and not list(p.glob(".*"))


def get_output_line():
    return "\n--------------------------------------------------------\n"


def read_info_file(data_dir: Path):
    info_file = data_dir / "data.info"
    series = pd.read_csv(info_file, sep=": ", index_col=0, header=None, engine="python").squeeze()
    return series.to_dict()


def save_to_csv(metric_list, output_file):
    out = pd.DataFrame(data=metric_list, columns=[*metric_list[0]._fields])
    out.to_csv(output_file, index=False)
