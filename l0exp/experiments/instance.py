import inspect
import l0learn
import numpy as np
import scipy.sparse as sparse
from numpy.typing import ArrayLike
from sklearn.metrics import f1_score
from el0ps.datafit import *  # noqa: F401, F403
from el0ps.penalty import *  # noqa: F401, F403


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
    calibration, datafit_name, penalty_name, A, y, x_true=None
):
    if calibration == "l0learn":
        return calibrate_parameters_l0learn(
            datafit_name, penalty_name, A, y, x_true
        )
    elif isinstance(calibration, dict):
        return calibrate_parameters_hardcoded(
            calibration, datafit_name, penalty_name, A, y, x_true
        )
    else:
        raise ValueError("Unknown calibration method: {}".format(calibration))


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


def calibrate_parameters_hardcoded(
    calibration, datafit_name, penalty_name, A, y, x_true=None
):
    datafit = get_datafit(datafit_name, y)
    penalty = get_penalty(penalty_name, **calibration["penalty"])
    lmbd = calibration["lmbd"]
    if x_true is None:
        if "x_true" in calibration:
            x_true = calibration["x_true"]
    return datafit, penalty, lmbd, x_true
