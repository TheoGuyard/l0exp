import numpy as np
import pickle
from numpy.typing import NDArray
from pathlib import Path


def load_dataset(dataset: str) -> tuple[NDArray, NDArray]:

    dataset_path = Path(dataset)
    dataset_suff = dataset_path.suffix

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file {dataset_path} not found.")

    if dataset_suff == ".npz":
        with open(dataset_path, "rb") as f:
            data = np.load(f)
        A = data["A"]
        y = data["y"]
    elif dataset_suff == ".pkl":
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        A = data["A"]
        y = data["y"]
    else:
        raise ValueError(f"Unsupported dataset suffix {dataset_suff}")

    return A, y
