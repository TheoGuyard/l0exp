import pathlib
import pickle
import numpy as np


dataset_dir = pathlib.Path(__file__).parent.parent / "datasets"
dataset_path = dataset_dir / "arcene.pkl"

with open(dataset_path, "rb") as f:
    data = pickle.load(f)
    print(data.keys())
    A = data["A"]
    y = data["y"]
    print(np.linalg.norm(A, axis=0))
    print(y)
    print(A.shape)

with open(dataset_dir / "new.pkl", "wb") as f:
    pickle.dump({"A": A, "y": y}, f)
    print("Saved new dataset")
