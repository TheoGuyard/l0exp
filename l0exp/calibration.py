import inspect
import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.conversion as roco
from copy import deepcopy
from itertools import product
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from scipy.sparse import csc_matrix
from sklearn.metrics import f1_score
from el0ps.datafit import BaseDatafit, Leastsquares
from el0ps.penalty import (
    BasePenalty,
    Bigm,
    BigmL1norm,
    BigmL2norm,
    L1norm,
    L2norm,
    PositiveL2norm,
    BigmPositiveL1norm,
    L1L2norm,
    Bounds
)
from el0ps.datafit import *  # noqa
from el0ps.penalty import *  # noqa
from el0ps.path import Path
from el0ps.solver import BnbSolver


def dgCMatrix_to_numpy(X):
    x = np.array(X.slots["x"])
    i = np.array(X.slots["i"])
    p = np.array(X.slots["p"])
    dim = tuple(np.array(X.slots["Dim"]))
    mat = csc_matrix((x, i, p), shape=dim)
    return np.array(mat.todense())


def get_calibration_hardcoded(
    A,
    y,
    x,
    datafit: str = "Leastsquares",
    penalty: str = "BigmL2norm",
    lmbd: float = 1.0,
    datafit_params: dict = {},
    penalty_params: dict = {},
):

    datafit_type = eval(datafit)
    datafit_sign = inspect.signature(datafit.__init__)
    datafit_args = [
        arg
        for arg in datafit_sign.parameters
        if arg not in ["self", "args", "kwargs"]
    ]

    penalty_type = eval(penalty)
    penalty_sign = inspect.signature(penalty_type.__init__)
    penalty_args = [
        arg
        for arg in penalty_sign.parameters
        if arg not in ["self", "args", "kwargs"]
    ]

    f = datafit_type(y, **{k: datafit_params[k] for k in datafit_args})
    h = penalty_type(**{k: penalty_params[k] for k in penalty_args})

    return f, h, lmbd


def get_calibration_mixture(
    A, y, x, distrib_name: str = "gaussian", distrib_args: dict = {}, **kwargs
):

    m, n = A.shape
    k = np.count_nonzero(x)
    w = A @ x
    s = np.linalg.norm(w) / np.linalg.norm(w - y)
    sigma = np.sqrt(np.sqrt((w @ w) / (m * s)))

    f = Leastsquares(y)

    if distrib_name == "gaussian":
        alpha = float(0.5 * (sigma / distrib_args["scale"]) ** 2)
        h = L2norm(alpha)
    elif distrib_name == "laplace":
        big_m = float(np.max(np.abs(x)))
        alpha = float(sigma**2 / distrib_args["scale"])
        h = BigmL1norm(M=big_m, alpha=alpha)
    elif distrib_name == "uniform":
        if distrib_args["low"] == -distrib_args["high"]:
            h = Bigm(distrib_args["high"])
        else:
            x_lb = distrib_args["low"] * np.ones(x.size)
            x_ub = distrib_args["high"] * np.ones(x.size)
            h = Bounds(x_lb=x_lb, x_ub=x_ub)
    elif distrib_name == "halfgaussian":
        alpha = float(0.5 * (sigma / distrib_args["scale"]) ** 2)
        h = PositiveL2norm(alpha)
    elif distrib_name in ["exponential", "halflaplace"]:
        big_m = float(np.max(np.abs(x)))
        alpha = float(sigma**2 / distrib_args["scale"])
        h = BigmPositiveL1norm(big_m, alpha)
    elif distrib_name == "gausslaplace":
        alpha = float(sigma**2 / distrib_args["scale1"])
        beta = float(0.5 * (sigma / distrib_args["scale2"]) ** 2)
        h = L1L2norm(alpha, beta)
    elif distrib_name == "dirac":
        h = Bigm(1.0)
    else:
        raise ValueError(
            f"Unknown distrib_name {distrib_name} for mixture calibration."
        )

    lmbd = float(sigma**2 * np.log((n - k) / k))

    return f, h, lmbd


def get_calibration_synthetic(*args, **kwargs):
    return get_calibration_mixture(
        *args, **kwargs, distrib_name="dirac", distrib_args={}
    )


def get_calibration_l0learn(A, y, x, datafit, penalty, **kwargs):

    bindings = {
        "Leastsquares": "SquaredError",
        "Logistic": "Logistic",
        "Squaredhinge": "SquaredHinge",
        "Bigm": "L0",
        "BigmL1norm": "L0L1",
        "BigmL2norm": "L0L2",
        "L1norm": "L0L1",
        "L2norm": "L0L2",
    }

    if datafit not in bindings.keys():
        raise ValueError(
            f"Datafit {datafit} not supported for l0learn calibration. "
            f"Available datafit and penalty bindings: {list(bindings.keys())}"
        )
    if penalty not in bindings.keys():
        raise ValueError(
            f"Penalty {penalty} not supported for l0learn calibration. "
            f"Available datafit and penalty bindings: {list(bindings.keys())}"
        )

    importr("L0Learn")

    with roco.localconverter(ro.default_converter + numpy2ri.converter):

        r_A = roco.py2rpy(A)
        r_y = roco.py2rpy(y)

        fit, cv_means, cv_stds = ro.r("L0Learn.cvfit")(
            x=r_A,
            y=r_y,
            loss=bindings[datafit],
            penalty=bindings[penalty],
            intercept=False,
            **kwargs,
        )

        fit = {str(k): v for (k, v) in zip(fit.names(), list(fit.values()))}
        cv_means = list(cv_means.values())
        cv_stds = list(cv_stds.values())

    f = eval(datafit)(y)

    best_M = None
    best_l = None
    best_a = None
    best_cv = np.inf
    best_f1 = 0.0
    for i, a in enumerate(fit["gamma"]):
        X = dgCMatrix_to_numpy(fit["beta"][i])
        for j, l in enumerate(fit["lambda"][i]):
            xj = X[:, j]
            cv = cv_means[i][j]
            f1 = 0.0 if x is None else f1_score(x != 0.0, xj != 0.0)
            if f1 >= best_f1:
                if cv < best_cv:
                    best_M = float(np.max(np.abs(xj)))
                    best_l = float(l)
                    best_a = float(a)
                    best_cv = cv
                    best_f1 = f1

    if penalty == "Bigm":
        h = Bigm(M=best_M)
    elif penalty == "BigmL1norm":
        h = BigmL1norm(M=best_M, alpha=best_a)
    elif penalty == "BigmL2norm":
        h = BigmL2norm(M=best_M, beta=best_a)
    elif penalty == "L1norm":
        h = L1norm(alpha=best_a)
    elif penalty == "L2norm":
        h = L2norm(beta=best_a)
    else:
        raise ValueError(f"Unknown penalty {penalty} for l0learn calibration.")

    return f, h, best_l


def get_calibration_cv(
    A,
    y,
    x,
    datafit,
    penalty,
    nfolds: int = 10,
    time_limit: float = 60.0,
    verbose: bool = False,
    **kwargs,
):

    m, n = A.shape

    f: BaseDatafit = eval(datafit)(y)

    M = np.max(np.abs(np.linalg.lstsq(A, y, rcond=None)[0]))

    grid_params = {}

    if "Bigm" in penalty:
        grid_params["M"] = M * np.array([1.0, 2.0, 5.0, 10.0])
    if "L1" in penalty:
        grid_params["alpha"] = m * np.logspace(-3, 0, 4)
    if "L2" in penalty:
        grid_params["beta"] = m * np.logspace(-3, 0, 4)

    grid_keys = list(grid_params.keys())
    grid_vals = list(grid_params.values())
    grid_params = [dict(zip(grid_keys, c)) for c in list(product(*grid_vals))]

    solver = BnbSolver(time_limit=time_limit, verbose=False)

    best_cvs = np.inf
    best_nnz = None
    best_p = None
    best_l = None

    if verbose:
        print("Starting CV calibration")

    for params in grid_params:

        if verbose:
            print(f"Testing parameters: {params}...")

        h: BasePenalty = eval(penalty)(**params)

        path = Path(**kwargs, verbose=verbose)
        results = path.fit(solver, f, h, A)

        for lmbd, result in results.items():

            nnz = np.count_nonzero(result.x)
            cvs = 0.0
            for _ in range(nfolds):
                fold = np.random.choice(m, m // nfolds, replace=False)
                A_fold = A[fold, :]
                y_fold = y[fold]
                f_fold = eval(datafit)(y_fold)
                cvs += f_fold.value(A_fold @ result.x)

            best_found = False
            if np.allclose(cvs, best_cvs, rtol=1e-4):
                if best_nnz is None or nnz < best_nnz:
                    best_found = True
            elif cvs < best_cvs:
                best_found = True

            if best_found:
                best_cvs = cvs
                best_nnz = nnz
                best_p = deepcopy(params)
                best_l = lmbd
                if verbose:
                    print(
                        f"  New best at lambda={best_l} with {nnz} nnz and "
                        f"cvs={best_cvs:.4f}"
                    )

    return f, eval(penalty)(**best_p), best_l


def get_calibration(A, y, x, type: str, args: dict):

    if type == "hardcoded":
        f, h, lmbd = get_calibration_hardcoded(A, y, x, **args)
    elif type == "mixture":
        f, h, lmbd = get_calibration_mixture(A, y, x, **args)
    elif type == "synthetic":
        f, h, lmbd = get_calibration_synthetic(A, y, x, **args)
    elif type == "l0learn":
        f, h, lmbd = get_calibration_l0learn(A, y, x, **args)
    elif type == "cv":
        f, h, lmbd = get_calibration_cv(A, y, x, **args)
    else:
        raise ValueError(f"Unknown problem calibration type {type}.")

    return f, h, lmbd
