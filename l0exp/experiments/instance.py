import inspect
import l0learn
import numpy as np
import scipy.sparse as sparse
from itertools import product
from numpy.typing import ArrayLike
from sklearn.metrics import f1_score
from el0ps.datafit import *  # noqa: F401, F403
from el0ps.penalty import *  # noqa: F401, F403
from el0ps.solver import BnbSolver
from el0ps.path import Path
from el0ps.utils import compute_lmbd_max

from l0exp.bigml1l2norm import BigmL1L2norm  # noqa: F401


def preprocess_data(
    A: ArrayLike,
    y: ArrayLike,
    x_true: ArrayLike,
    center: bool = False,
    normalize: bool = False,
    y_binary: bool = False,
) -> list:
    """Pre-process problem data."""
    if sparse.issparse(A):
        A = A.todense()
    if not A.flags["F_CONTIGUOUS"] or not A.flags["OWNDATA"]:
        A = np.array(A, order="F")
    zero_columns = np.abs(np.linalg.norm(A, axis=0)) < 1e-7
    if np.any(zero_columns):
        A = np.array(A[:, np.logical_not(zero_columns)], order="F")
    if center:
        A -= np.mean(A, axis=0)
        y -= np.mean(y)
    if normalize:
        A /= np.linalg.norm(A, axis=0, ord=2)
        y /= np.linalg.norm(y, ord=2)
    if y_binary:
        y_cls = np.unique(y)
        assert y_cls.size == 2
        y_cls0 = y == y_cls[0]
        y_cls1 = y == y_cls[1]
        y = np.zeros(y.size, dtype=float)
        y[y_cls0] = -1.0
        y[y_cls1] = 1.0
    return A, y, x_true


def calibrate_parameters(
    method, datafit_name, penalty_name, A, y, x_true=None, **kwargs
):
    if method == "l0learn":
        calibration = calibrate_parameters_l0learn
    elif method == "cv":
        calibration = calibrate_parameters_cv
    elif method == "hardcoded":
        calibration = calibrate_parameters_hardcoded
    else:
        raise ValueError("Unknown calibration method: {}".format(method))

    return calibration(datafit_name, penalty_name, A, y, x_true, **kwargs)


def get_datafit(datafit_name, y):
    return eval(datafit_name)(y)


def get_penalty(penalty_name, **kwargs):
    penalty_type = eval(penalty_name)
    penalty_sign = inspect.signature(penalty_type.__init__)
    penalty_args = {
        name
        for name, param in penalty_sign.parameters.items()
        if name != "self"
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    kwargs_filtered = {k: v for k, v in kwargs.items() if k in penalty_args}
    penalty = penalty_type(**kwargs_filtered)
    return penalty


def calibrate_parameters_l0learn(
    datafit_name, penalty_name, A, y, x_true=None
):
    """Give some problem data A and y, datafit and penalty, use `l0learn` to
    find an appropriate L0-norm weight and suitable hyperparameters for the
    penalty function."""

    # Binding for datafit and penalty names between El0ps and L0learn
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

    assert datafit_name in bindings.keys()
    assert penalty_name in bindings.keys()

    m, n = A.shape

    # Datafit instanciation
    datafit = get_datafit(datafit_name, y)

    # Fit an approximate regularization path with L0Learn
    cvfit = l0learn.cvfit(
        A,
        y,
        bindings[datafit_name],
        bindings[penalty_name],
        intercept=False,
        num_folds=5,
    )

    # Penalty and L0-norm parameters calibration from L0learn path. Select the
    # hyperparameters with the best cross-validation score among those with the
    # best support recovery F1 score.
    best_M = None
    best_lmbda = None
    best_gamma = None
    best_cv = np.inf
    best_f1 = 0.0
    best_x = None
    for i, gamma in enumerate(cvfit.gamma):
        for j, lmbda in enumerate(cvfit.lambda_0[i]):
            x = cvfit.coeff(lmbda, gamma, include_intercept=False)
            x = np.array(x.todense()).reshape(n)
            cv = cvfit.cv_means[i][j][0]
            f1 = 0.0 if x_true is None else f1_score(x_true != 0.0, x != 0.0)
            if (f1 > best_f1) or (x_true is None):
                if cv < best_cv:
                    best_M = np.max(np.abs(x))
                    best_lmbda = lmbda
                    best_gamma = gamma
                    best_cv = cv
                    best_f1 = f1
                    best_x = np.copy(x)

    # Penalty instantiation
    penalty = get_penalty(
        penalty_name, M=best_M, alpha=best_gamma, beta=best_gamma
    )

    return datafit, penalty, best_lmbda, best_x


def calibrate_parameters_cv(
    datafit_name,
    penalty_name,
    A,
    y,
    x_true=None,
    criterion: str = "bic",
    time_limit: float = 60.0,
    **kwargs,
):

    m, n = A.shape

    datafit = get_datafit(datafit_name, y)

    if datafit_name == "Leastsquares":
        scaling = 2.0 * m
    elif datafit_name == "Logistic":
        scaling = 1.0 * m
    elif datafit_name == "Squaredhinge":
        scaling = 1.0 * m
    else:
        raise ValueError(f"Unknown datafit name: {datafit_name}")

    x_lstsq = np.linalg.lstsq(A, y, rcond=None)[0]
    M = np.max(np.abs(x_lstsq))

    grid_params = {}

    if penalty_name == "Bigm":
        grid_params["M"] = [M, 10.0 * M, 100.0 * M, 1000.0 * M]
    elif penalty_name == "BigmL1norm":
        grid_params["M"] = [M, 10.0 * M, 100.0 * M, 1000.0 * M]
        grid_params["alpha"] = [0.1 * m, 1.0 * m, 10.0 * m, 100.0 * m]
    elif penalty_name == "BigmL2norm":
        grid_params["M"] = [M, 10.0 * M, 100.0 * M, 1000.0 * M]
        grid_params["beta"] = [0.1 * m, 1.0 * m, 10.0 * m, 100.0 * m]
    elif penalty_name == "BigmL1L2norm":
        grid_params["M"] = [M, 10.0 * M, 100.0 * M, 1000.0 * M]
        grid_params["alpha"] = [0.001 * m, 0.01 * m, 0.1 * m, 1.0 * m]
        grid_params["beta"] = [0.1 * m, 1.0 * m, 10.0 * m, 100.0 * m]
    elif penalty_name == "L1L2norm":
        grid_params["alpha"] = [0.1 * m, 1.0 * m, 10.0 * m, 100.0 * m]
        grid_params["beta"] = [0.1 * m, 1.0 * m, 10.0 * m, 100.0 * m]
    else:
        raise ValueError(f"Unknown penalty name: {penalty_name}")

    grid_keys = list(grid_params.keys())
    grid_vals = list(grid_params.values())
    grid_params = [dict(zip(grid_keys, c)) for c in list(product(*grid_vals))]

    solver = BnbSolver(time_limit=time_limit)

    best_found = False
    best_criterion = None
    best_params = None
    best_loss = None
    best_nnnz = None
    best_lmbd = None
    best_lratio = None
    best_x = None

    for params in grid_params:
        print(f"Params: {params}")
        penalty = get_penalty(penalty_name, **params)
        lmbdmax = compute_lmbd_max(datafit, penalty, A)
        path = Path(**kwargs)
        results = path.fit(solver, datafit, penalty, A)

        for lmbd, result in results.items():

            loss = datafit.value(A @ result.x)
            nnnz = np.count_nonzero(result.x)

            if criterion == "aic":
                criterion_value = 2.0 * scaling * loss + 2.0 * nnnz
            elif criterion == "bic":
                criterion_value = 2.0 * scaling * loss + np.log(m) * nnnz
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

            if best_criterion is None:
                best_found = True
            else:
                if np.abs(criterion_value - best_criterion) < 1e-5:
                    if lmbd > best_lmbd:
                        best_found = True
                elif criterion_value < best_criterion:
                    best_found = True

            if best_found:
                best_found = False
                best_criterion = criterion_value
                best_params = params
                best_loss = loss
                best_nnnz = nnnz
                best_lmbd = lmbd
                best_lratio = lmbd / lmbdmax
                best_x = np.copy(result.x)
                print("Found new best criterion")
                print(f"  {criterion}\t: {best_criterion}")
                print(f"  lambda: {best_lmbd}")
                print(f"  lratio: {best_lratio}")
                print(f"  loss  : {best_loss}")
                print(f"  nnnz  : {best_nnnz}")

    print()
    print("Overall best criterion")
    print(f"  {criterion}\t: {best_criterion}")
    print(f"  lambda: {best_lmbd}")
    print(f"  lratio: {best_lratio}")
    print(f"  loss  : {best_loss}")
    print(f"  nnnz  : {best_nnnz}")

    return datafit, get_penalty(penalty_name, **best_params), best_lmbd, best_x


def calibrate_parameters_hardcoded(
    datafit_name,
    penalty_name,
    A,
    y,
    x_true=None,
    penalty_params=None,
    lmbd=None,
    x_cal=None,
):
    datafit = get_datafit(datafit_name, y)
    penalty = get_penalty(penalty_name, **penalty_params)
    return datafit, penalty, lmbd, x_cal
