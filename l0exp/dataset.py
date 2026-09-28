import numpy as np
import openml
import os
import pathlib
import pickle
import tempfile
import urllib.request
import zipfile
from libsvmdata import fetch_libsvm
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
from scipy.special import erfc
from typing import Optional
from numpy.typing import NDArray

# Mixture datasets helpers


def gauss_laplace_pdf(x, theta):
    scale1 = theta[0]
    scale2 = theta[1]
    c1 = np.sqrt(0.5 / (np.pi * scale2**2))
    c2 = erfc(1.0 / (scale1 * np.sqrt(2.0 / scale2**2)))
    c3 = (
        (1.0 / scale1) * np.abs(x)
        + 0.5 * (x / scale2) ** 2
        + 0.5 * (scale2 / scale1) ** 2
    )
    return (c1 / c2) * np.exp(-c3)


def inverse_transform_sampling(pdf, theta, lb, ub, size=1, n=1000):
    xs = np.linspace(lb, ub, n)
    cs = np.array([quad(lambda t: pdf(t, theta), lb, x)[0] for x in xs])
    cs /= cs[-1]
    ic = interp1d(cs, xs, bounds_error=False, fill_value=(lb, ub))
    v = np.random.uniform(0, 1, size)
    return ic(v)[0]


def sample_distribution(name: str, args: dict = {}):
    if name == "dirac":
        u = 1.0
    elif name in ["gaussian", "normal"]:
        u = np.random.normal(0.0, args["scale"])
    elif name == "laplace":
        u = np.random.laplace(0.0, args["scale"])
    elif name == "uniform":
        assert args["low"] <= 0.0 <= args["high"]
        u = np.random.uniform(args["low"], args["high"])
    elif name in ["halfgaussian", "halfnormal"]:
        u = np.abs(np.random.normal(0.0, args["scale"]))
    elif name in ["exponential", "halflaplace"]:
        u = np.abs(np.random.laplace(0.0, args["scale"]))
    elif name == "gausslaplace":
        u = inverse_transform_sampling(
            gauss_laplace_pdf,
            (args["scale1"], args["scale2"]),
            -2.326 / args["scale2"] ** 2,  # 0.01 normal quantile
            2.326 / args["scale2"] ** 2,  # 0.99 normal quantile
        )
    else:
        raise ValueError(f"Unknown distribution name {name}.")
    return u


# Realworld datasets helpers


def load_matrix_dense(file_path: str, **kwargs) -> np.ndarray:
    matrix = np.loadtxt(file_path, **kwargs)
    return matrix


def load_matrix_sparse(file_path: str) -> np.ndarray:
    rows = []
    cols = []
    cdim = 0
    with open(file_path) as f:
        for i, line in enumerate(f):
            if line.strip():
                indices = []
                values = []
                for item in line.strip().split():
                    index, value = item.split(":")
                    indices.append(int(index))
                    values.append(float(value))
                rows.extend([i] * len(indices))
                cols.extend(indices)
                cdim = max(cdim, max(indices))
    data = np.ones(len(rows), dtype=np.int8)
    matrix = csr_matrix(
        (data, (rows, cols)), shape=(i + 1, cdim + 1)
    ).todense()
    return matrix


def load_matrix_sparse_binary(file_path: str) -> np.ndarray:
    rows = []
    cols = []
    cdim = 0
    with open(file_path) as f:
        for i, line in enumerate(f):
            if line.strip():
                j = np.fromstring(line, sep=" ", dtype=int)
                rows.extend([i] * len(j))
                cols.extend(j)
                cdim = max(cdim, max(j))
    data = np.ones(len(rows), dtype=np.int8)
    matrix = csr_matrix(
        (data, (rows, cols)), shape=(i + 1, cdim + 1)
    ).todense()
    return matrix


def load_dataset_drug():
    dataset = openml.datasets.get_dataset(dataset_id=46137)
    A, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    A = A.to_numpy()
    y = y.to_numpy()
    return A, y, "regression"


def load_dataset_kits():
    dataset = openml.datasets.get_dataset(dataset_id=42764)
    A, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    A = A.to_numpy()
    y = y.to_numpy()
    return A, y, "regression"


def load_dataset_riboflavin():
    dataset = openml.datasets.get_dataset(dataset_id=46983)
    A, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    A = A.to_numpy()
    y = y.to_numpy()
    return A, y, "regression"


def load_dataset_arcene():

    url = "https://archive.ics.uci.edu/static/public/167/arcene.zip"
    A_path = "ARCENE/arcene_train.data"
    y_path = "ARCENE/arcene_train.labels"

    with tempfile.TemporaryDirectory() as tmpdirname:

        zip_path = os.path.join(tmpdirname, "arcene.zip")
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdirname)

        A_path = os.path.join(tmpdirname, A_path)
        y_path = os.path.join(tmpdirname, y_path)

        A = load_matrix_dense(A_path)
        y = load_matrix_dense(y_path)

    return A, y, "classification"


def load_dataset_breast_cancer():
    A, y = fetch_libsvm("duke breast-cancer")
    return A, y, "classification"


def load_dataset_colon_cancer():
    A, y = fetch_libsvm("colon-cancer")
    return A, y, "classification"


def load_dataset_dexter():

    url = "https://archive.ics.uci.edu/static/public/168/dexter.zip"
    A_path = "DEXTER/dexter_train.data"
    y_path = "DEXTER/dexter_train.labels"

    with tempfile.TemporaryDirectory() as tmpdirname:

        zip_path = os.path.join(tmpdirname, "dexter.zip")
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdirname)

        A_path = os.path.join(tmpdirname, A_path)
        y_path = os.path.join(tmpdirname, y_path)

        A = load_matrix_sparse(A_path)
        y = load_matrix_dense(y_path)

    return A, y, "classification"


def load_dataset_dorothea():

    url = "https://archive.ics.uci.edu/static/public/169/dorothea.zip"
    A_path = "DOROTHEA/dorothea_train.data"
    y_path = "DOROTHEA/dorothea_train.labels"

    with tempfile.TemporaryDirectory() as tmpdirname:

        zip_path = os.path.join(tmpdirname, "dorothea.zip")
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdirname)

        A_path = os.path.join(tmpdirname, A_path)
        y_path = os.path.join(tmpdirname, y_path)

        A = load_matrix_sparse_binary(A_path)
        y = load_matrix_dense(y_path)

    return A, y, "classification"


def load_dataset_gisette():

    url = "https://archive.ics.uci.edu/static/public/170/gisette.zip"
    A_path = "GISETTE/gisette_train.data"
    y_path = "GISETTE/gisette_train.labels"

    with tempfile.TemporaryDirectory() as tmpdirname:

        zip_path = os.path.join(tmpdirname, "gisette.zip")
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdirname)

        A_path = os.path.join(tmpdirname, A_path)
        y_path = os.path.join(tmpdirname, y_path)

        A = load_matrix_dense(A_path)
        y = load_matrix_dense(y_path)

    return A, y, "classification"


def load_dataset_leukemia():
    A, y = fetch_libsvm("leukemia")
    return A, y, "classification"


def load_dataset_madelon():

    url = "https://archive.ics.uci.edu/static/public/171/madelon.zip"
    A_path = "MADELON/madelon_train.data"
    y_path = "MADELON/madelon_train.labels"
    with tempfile.TemporaryDirectory() as tmpdirname:

        zip_path = os.path.join(tmpdirname, "madelon.zip")
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdirname)

        A_path = os.path.join(tmpdirname, A_path)
        y_path = os.path.join(tmpdirname, y_path)

        A = load_matrix_sparse_binary(A_path)
        y = load_matrix_dense(y_path)

    return A, y, "classification"


# Dataset loaders


def get_dataset_mixture(
    k: int = 10,
    m: int = 500,
    n: int = 1000,
    r: float = 0.5,
    s: float = 10.0,
    distrib_name: str = "gaussian",
    distrib_args: dict = {},
    seed=None,
):

    assert n >= k > 0
    assert m > 0
    assert 0.0 <= r < 1.0
    assert s > 0.0

    if seed is not None:
        np.random.seed(seed)

    # Ground truth
    x = np.zeros(n)
    for i in np.linspace(0, n - 1, num=k, dtype=int):
        x[i] = sample_distribution(distrib_name, distrib_args)

    # Design matrix
    M = np.zeros(n)
    N1 = np.repeat(np.arange(n).reshape(n, 1), n).reshape(n, n)
    N2 = np.repeat(np.arange(n).reshape(1, n), n).reshape(n, n).T
    K = np.power(r, np.abs(N1 - N2))
    A = np.random.multivariate_normal(M, K, size=m)
    A /= np.linalg.norm(A, axis=0, ord=2)

    # Noise vector
    w = A @ x
    e = np.random.normal(0.0, np.sqrt((w @ w) / (m * s)), m)

    # Observation vector
    y = w + e

    return A, y, x


def get_dataset_synthetic(*args, **kwargs):
    return get_dataset_mixture(
        *args, **kwargs, distrib_name="dirac", distrib_args={}
    )


def get_dataset_realworld(name: str, normalize: bool = True):

    available = [
        # regression
        "drug",
        "kits",
        "riboflavin",
        # classification
        "arcene",
        "breast-cancer",
        "colon-cancer",
        "dexter",
        "dorothea",
        "gisette",
        "leukemia",
        "madelon",
    ]

    if name == "drug":
        A, y, task = load_dataset_drug()
    elif name == "kits":
        A, y, task = load_dataset_kits()
    elif name == "riboflavin":
        A, y, task = load_dataset_riboflavin()
    #
    elif name == "arcene":
        A, y, task = load_dataset_arcene()
    elif name == "breast-cancer":
        A, y, task = load_dataset_breast_cancer()
    elif name == "colon-cancer":
        A, y, task = load_dataset_colon_cancer()
    elif name == "dexter":
        A, y, task = load_dataset_dexter()
    elif name == "dorothea":
        A, y, task = load_dataset_dorothea()
    elif name == "gisette":
        A, y, task = load_dataset_gisette()
    elif name == "leukemia":
        A, y, task = load_dataset_leukemia()
    elif name == "madelon":
        A, y, task = load_dataset_madelon()

    else:
        raise ValueError(
            f"Unknown realworld dataset name {name}. Available names are: "
            f"{available}"
        )

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).flatten()

    col_nnz = np.where(np.linalg.norm(A, axis=0, ord=2) != 0.0)[0]
    if col_nnz.size < A.shape[1]:
        A = A[:, col_nnz]

    if normalize:
        A /= np.linalg.norm(A, axis=0, ord=2)

    if task == "classification":
        if set(np.unique(y)) != {1.0, -1.0}:
            raise ValueError(
                f"Unexpected labels in classification dataset {name}."
            )

    return A, y, None


def get_dataset_hardcoded(name: str, normalize: bool = True):

    datasets_dir = pathlib.Path(__file__).parent / "datasets"
    dataset_path = datasets_dir / f"{name}.pkl"

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
        A, y, x, task = data["A"], data["y"], data["x"], data["task"]

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).flatten()
    x = np.asarray(x, dtype=float).flatten() if x is not None else None

    col_nnz = np.where(np.linalg.norm(A, axis=0, ord=2) != 0.0)[0]
    if col_nnz.size < A.shape[1]:
        A = A[:, col_nnz]

    if normalize:
        A /= np.linalg.norm(A, axis=0, ord=2)

    if task == "classification":
        if set(np.unique(y)) != {1.0, -1.0}:
            raise ValueError(
                f"Unexpected labels in classification dataset {name}."
            )

    return A, y, x


def get_dataset(
    type: dict, args: dict
) -> tuple[NDArray, NDArray, Optional[NDArray]]:
    if type == "synthetic":
        return get_dataset_synthetic(**args)
    elif type == "mixture":
        return get_dataset_mixture(**args)
    elif type == "realworld":
        return get_dataset_realworld(**args)
    elif type == "hardcoded":
        return get_dataset_hardcoded(**args)
    else:
        raise ValueError(f"Unknown dataset type {type}.")
