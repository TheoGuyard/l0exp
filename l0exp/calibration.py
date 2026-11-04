import inspect
import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.conversion as roco
from itertools import product
from numba import njit
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from scipy.sparse import csc_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from el0ps.datafit import Leastsquares
from el0ps.penalty import (
    Bigm,
    BigmL1norm,
    BigmL2norm,
    L1norm,
    L2norm,
    PositiveL2norm,
    BigmPositiveL1norm,
    L1L2norm,
    Bounds,
)
from el0ps.datafit import *  # noqa
from el0ps.penalty import *  # noqa
from el0ps.compilation import compiled_clone
from el0ps.solver import BoundSolver
from el0ps.utils import compute_lmbd_max


@njit
def _solve_omp(f, A, h, lmbd, max_iter=1_000, tol=1e-4):
    """Greedy forward selection until the objective cannot decrease."""
    m, n = A.shape
    x = np.zeros(n)
    w = np.zeros(m)  # w = A @ x
    v = f.value(w)
    support = np.zeros(0, dtype=np.int64)
    visited = np.zeros(n, dtype=np.bool_)
    g = np.abs(A.T @ f.gradient(w))
    for _ in range(min(m, n)):
        # Select next index to add to the support
        j = np.argmax(g)
        if g[j] <= 0.0:
            break
        visited[j] = True
        g[j] = -np.inf
        support_new = np.append(support, j)

        # Recover solution on new support by minimizing f(Ax) + h(x)
        A_new = np.ascontiguousarray(A[:, support_new])
        lip = f.gradient_lipschitz_constant() * np.linalg.norm(A_new, 2) ** 2
        eta = 1.0 / lip
        x_sub = np.copy(x[support_new])
        z = x_sub.copy()
        t = 1.0
        for _ in range(max_iter):
            x_old = x_sub
            # Gradient step over f(Ax)
            u = z - eta * (A_new.T @ f.gradient(A_new @ z))
            # Proximal step over eta * h(x)
            x_sub = np.empty_like(u)
            for k in range(u.size):
                x_sub[k] = h.prox(support_new[k], u[k], eta)
            # Momentum step
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
            z = x_sub + ((t - 1.0) / t_new) * (x_sub - x_old)
            t = t_new
            # Stationary check
            if np.linalg.norm(x_sub - x_old) <= tol * max(
                1.0, np.linalg.norm(x_sub)
            ):
                break

        # Discard j if it is not used by the solution
        if x_sub[-1] == 0.0:
            continue
        w_new = A_new @ x_sub
        v_new = f.value(w_new)
        for k in range(support_new.size):
            v_new += h.value(support_new[k], x_sub[k])

        # Non-decreasing check
        if v - v_new <= lmbd:
            break

        support, w, v = support_new, w_new, v_new
        x[support] = x_sub
        g = np.abs(A.T @ f.gradient(w))
        g[visited] = -np.inf

    return x


@njit
def _solve_iht(f, A, h, lmbd, max_iter=1_000, tol=1e-4):
    """Iterative hard thresholding heuristic over the L0-problem, with a step
    size selected by backtracking line search."""
    # Initial step size from the per-coordinate curvature of f(Ax)
    lip = f.gradient_lipschitz_constant() * np.max(np.sum(A**2, axis=0))
    eta = 1.0 / lip
    x = np.zeros(A.shape[1])
    w = A @ x
    fw = f.value(w)
    for _ in range(max_iter):
        g = A.T @ f.gradient(w)
        while True:
            # Gradient step over f(Ax)
            u = x - eta * g
            # Thresholding step over g(x) = eta * (lmbd * ||x||_0 + h(x))
            x_new = np.zeros_like(u)
            for i in range(u.size):
                v = h.prox(i, u[i], eta)
                r = (0.5 / eta) * (u[i] ** 2 - (u[i] - v) ** 2)
                if r - h.value(i, v) > lmbd:
                    x_new[i] = v
            # Sufficient decrease check
            d = x_new - x
            w_new = A @ x_new
            fw_new = f.value(w_new)
            if fw_new <= fw + g @ d + (0.5 / eta) * (d @ d):
                break
            eta *= 0.5
        x_old = x
        x, w, fw = x_new, w_new, fw_new
        # Stationary check
        if np.linalg.norm(x - x_old) <= tol * max(1.0, np.linalg.norm(x)):
            break
    return x


def _solve_cvx(f, A, h, lmbd, max_iter=1_000, tol=1e-4):
    """Convex relaxation of the L0-problem, solved with el0ps' BoundSolver at
    the root node of the Branch-and-Bound tree (all entries relaxed)."""
    n = A.shape[1]
    A = np.asfortranarray(A)
    solver = compiled_clone(BoundSolver(iter_limit=max_iter, relative_tol=tol))
    solver.setup(f, h, A, lmbd)
    solver.bound(
        f,
        h,
        A,
        lmbd,
        np.zeros(n, dtype=np.bool_),  # S0
        np.zeros(n, dtype=np.bool_),  # S1
        np.ones(n, dtype=np.bool_),  # Sb
        np.zeros(n),  # x
        np.inf,  # ub
        False,  # upper
    )
    return solver.x


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
    **kwargs,
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
    method: str = "cvx",
    nfolds: int = 10,
    grid_values: list = np.logspace(0, -4, 11),
    lmbd_values: list = np.logspace(0, -4, 11),
    factor: float = 1.5,
    seed: int = None,
    verbose: bool = False,
    **kwargs,
):
    """Calibrate the problem hyperparameters by k-fold cross-validation.

    The penalty hyperparameters alpha and beta (when present) are searched
    over the grid m * ``grid_values`` and the L0-norm weight lmbd over the
    grid m * ``lmbd_values``, where m is the number of samples. For each grid
    point and each training fold, an approximate solution of the
    L0-regularized problem is computed with a ``method`` ("omp", "iht" or "cvx")
    and the datafit value on the held-out fold is averaged over folds. The
    penalty hyperparameters with the smallest cross-validation error over all
    values of lmbd are selected, along with the corresponding lmbd.

    The Big-M constraint is discarded during the calibration and is set a
    posteriori as ``M = factor * ||xhat||_inf`` where ``xhat`` is the
    approximate solution obtained on the full dataset with the selected
    hyperparameters.
    """

    MAP_METHODS = {"omp": _solve_omp, "iht": _solve_iht, "cvx": _solve_cvx}

    penalty_type = eval(penalty)
    penalty_args = set(inspect.signature(penalty_type.__init__).parameters)
    penalty_args.remove("self")
    if not set(penalty_args).issubset({"M", "alpha", "beta"}):
        raise ValueError(f"Unsupported {penalty} for cv calibration.")

    solve = MAP_METHODS[method]
    A = np.ascontiguousarray(A, dtype=float)
    m, _ = A.shape
    f = eval(datafit)(y)

    def get_penalty(params, num_samples):
        kwargs = {k: num_samples * v for k, v in params.items()}
        if "M" in penalty_args:
            kwargs["M"] = np.inf  # calibrated a posteriori
        return penalty_type(**kwargs)

    # Hyperparameter grid (relative to the number of samples)
    keys = [k for k in ["alpha", "beta"] if k in penalty_args]
    grid = [dict(zip(keys, v)) for v in product(grid_values, repeat=len(keys))]

    if verbose:
        print(f"Starting CV calibration...")

    # Cross-validation loop
    fd = list(KFold(nfolds, shuffle=True, random_state=seed).split(A))
    cv = np.zeros((len(grid), len(lmbd_values)))
    for p, params in enumerate(grid):
        for train, test in fd:
            # Discard features that are all-zero over the training fold
            cols = np.flatnonzero(np.any(A[train] != 0.0, axis=0))
            A_train = np.ascontiguousarray(A[np.ix_(train, cols)])
            A_test = A[np.ix_(test, cols)]
            f_train = eval(datafit)(y[train])
            f_test = eval(datafit)(y[test])
            h = get_penalty(params, train.size)
            lmbd_max = compute_lmbd_max(f_train, h, A_train)
            f_train_jit, h_jit = compiled_clone(f_train), compiled_clone(h)
            for l, lmbd in enumerate(lmbd_max * np.asarray(lmbd_values)):
                x = solve(f_train_jit, A_train, h_jit, lmbd, **kwargs)
                cv[p, l] += f_test.value(A_test @ x) / test.size / nfolds
        if verbose:
            l = np.argmin(cv[p])
            print(
                f"  cv={cv[p, l]:.4e} at lmbd={lmbd_values[l]:.2e}*lmbd_max "
                f"for params {params}"
            )

    # Select the hyperparameters with the best cv error over all lmbd values
    p, l = np.unravel_index(np.argmin(cv), cv.shape)
    h = get_penalty(grid[p], m)
    lmbd = compute_lmbd_max(f, h, A) * lmbd_values[l]
    params = h.params_to_dict()

    # Valibrate big-M a posteriori (re-fit over full dataset)
    if "M" in penalty_args:
        xhat = solve(compiled_clone(f), A, compiled_clone(h), lmbd, **kwargs)
        params["M"] = factor * np.max(np.abs(xhat))
    params = {k: float(v) for k, v in params.items()}

    if verbose:
        print(
            f"Select params {params} with cv={cv[p,l]:.4e} at lmbd={lmbd:.4e}"
        )

    return f, penalty_type(**params), float(lmbd)


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
