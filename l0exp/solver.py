import gurobipy as gp
import numpy as np
import os
import pathlib
import re
import shutil
import subprocess
import sys
from gurobipy import GRB, nlfunc
from numpy.typing import NDArray
from typing import Optional, Union
from el0ps.datafit import (
    BaseDatafit,
    Leastsquares,
    Logistic,
    Squaredhinge,
    MipDatafit,
)
from el0ps.penalty import (
    BasePenalty,
    Bigm,
    BigmL1L2norm,
    BigmL1norm,
    BigmL2norm,
    BigmPositiveL1norm,
    BigmPositiveL2norm,
    Bounds,
    L1L2norm,
    L2norm,
    PositiveL2norm,
    MipPenalty,
)
from el0ps.datafit import *  # noqa
from el0ps.penalty import *  # noqa
from el0ps.solver import (
    BaseSolver,
    Result,
    Status,
    BnbSolver,
    MipSolver,
    OaSolver,
)
from el0ps.solver.mip import _mip_supports
from el0ps.path import Path
from el0ps.compilation import compiled_clone
from el0ps.utils import compute_lmbd_max
from l0bnb import BNBTree


class El0psSolver(BnbSolver):
    pass


class L0bnbSolver(BaseSolver):

    def __init__(
        self,
        integrality_tol: float = 0.0,
        relative_gap: float = 1e-8,
        absolute_gap: float = 0.0,
        time_limit: float = float(sys.maxsize),
        verbose: bool = False,
    ):
        self.integrality_tol = integrality_tol
        self.relative_gap = relative_gap
        self.absolute_gap = absolute_gap
        self.time_limit = time_limit
        self.verbose = verbose

    def __str__(self):
        return "L0bnbSolver"

    def solve(
        self,
        datafit: Leastsquares,
        penalty: Union[Bigm, L2norm, BigmL2norm],
        A: NDArray,
        lmbd: float,
        x_init: Union[NDArray, None] = None,
    ) -> Result:

        assert isinstance(datafit, Leastsquares)
        assert (
            isinstance(penalty, Bigm)
            or isinstance(penalty, L2norm)
            or isinstance(penalty, BigmL2norm)
        )

        if isinstance(penalty, Bigm):
            l0 = lmbd
            l2 = 0.0
            M = penalty.M
        elif isinstance(penalty, L2norm):
            l0 = lmbd
            l2 = penalty.beta
            M = sys.maxsize
        elif isinstance(penalty, BigmL2norm):
            l0 = lmbd
            l2 = penalty.beta
            M = penalty.M

        solver = BNBTree(
            A,
            datafit.y,
            self.integrality_tol,
            self.relative_gap,
        )

        result = solver.solve(
            l0,
            l2,
            M,
            gap_tol=self.relative_gap,
            warm_start=x_init,
            verbose=self.verbose,
            time_limit=self.time_limit,
        )

        if result.sol_time < self.time_limit:
            status = Status.OPTIMAL
        else:
            status = Status.TIME_LIMIT

        solution = np.array(result.beta)

        objective_value = (
            datafit.value(A @ solution)
            + lmbd * np.linalg.norm(solution, ord=0)
            + sum(penalty.value(i, xi) for i, xi in enumerate(solution))
        )

        return Result(
            status,
            result.sol_time,
            solver.number_of_nodes,
            solution,
            objective_value,
            None,
        )


class MimosaSolver(BaseSolver):

    def __init__(
        self,
        integrality_tol: float = 0.0,
        relative_gap: float = 1e-8,
        absolute_gap: float = 0.0,
        time_limit: float = float(sys.maxsize),
        verbose: bool = False,
    ):
        self.integrality_tol = integrality_tol
        self.relative_gap = relative_gap
        self.absolute_gap = absolute_gap
        self.time_limit = time_limit
        self.verbose = verbose

        if "MIMOSA_BIN" not in os.environ:
            raise ValueError("MIMOSA_BIN environment variable is not set.")
        self.MIMOSA_BIN = pathlib.Path(os.environ["MIMOSA_BIN"]).absolute()

        if "MIMOSA_TMP" not in os.environ:
            raise ValueError("MIMOSA_TMP environment variable is not set.")
        self.MIMOSA_TMP = pathlib.Path(os.environ["MIMOSA_TMP"]).absolute()

    def __str__(self):
        return "MimosaSolver"

    def solve(
        self,
        datafit: Leastsquares,
        penalty: Bigm,
        A: NDArray,
        lmbd: float,
        x_init: Union[NDArray, None] = None,
    ) -> Result:

        assert isinstance(datafit, Leastsquares)
        assert isinstance(penalty, Bigm)

        # Clean and create MIMOSA_TMP
        if self.MIMOSA_TMP.is_dir():
            shutil.rmtree(self.MIMOSA_TMP)
        self.MIMOSA_TMP.mkdir()

        # Save instance to MIMOSA_TMP
        np.savetxt(self.MIMOSA_TMP / "A.dat", A)
        np.savetxt(self.MIMOSA_TMP / "y.dat", datafit.y)
        np.savetxt(self.MIMOSA_TMP / "mu.dat", [lmbd])

        # Mimosa command
        options = "l2pl0 bb_activeset_warm 0 0.0 0 heap_on_lb 0 max_xi"
        results = self.MIMOSA_TMP / "results"
        command = "echo {} | {} {} {}.csv {} | tee {}.log".format(
            self.MIMOSA_TMP,
            self.MIMOSA_BIN,
            options,
            results,
            self.time_limit,
            results,
        )

        # Run Mimosa
        subprocess.run(
            command,
            shell=True,
            stdout=subprocess.STDOUT if self.verbose else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if self.verbose else subprocess.DEVNULL,
        )

        # Recover results
        output = results.with_suffix(".log")
        if output.is_file():
            with open(output, "r") as f:
                content = f.read()
            status = Status.OPTIMAL
            match = re.search(r"temps_d'execution:\s*([\d.]+)", content)
            solve_time = float(match.group(1)) if match else None
            match = re.search(r"Node_Number_BB:\s*(\d+)", content)
            iter_count = int(match.group(1)) if match else None
            match = re.search(
                r"x_sol\s*((?:\s*-?\d*\.?\d+\s*)+)", content, re.DOTALL
            )  # noqa: E501
            if match:
                x_lines = match.group(1)
                x_vals = list(map(float, re.findall(r"-?\d*\.?\d+", x_lines)))
                x = np.array(x_vals)
            else:
                x = np.zeros(A.shape[1])
            if status == Status.OPTIMAL:
                x = np.clip(x, -penalty.M, penalty.M)
            objective_value = (
                datafit.value(A @ x)
                + lmbd * np.linalg.norm(x, ord=0)
                + sum(penalty.value(i, xi) for i, xi in enumerate(x))
            )
        else:
            status = Status.UNKNOWN
            solve_time = np.nan
            iter_count = -1
            x = np.zeros(A.shape[1])
            objective_value = np.nan

        # Clean MIMOSA_TMP
        if self.MIMOSA_TMP.is_dir():
            shutil.rmtree(self.MIMOSA_TMP)

        return Result(
            status,
            solve_time,
            iter_count,
            x,
            objective_value,
            None,
        )


class GurobiSolver(BaseSolver):

    def __init__(
        self,
        relative_gap: float = 1e-8,
        absolute_gap: float = 0.0,
        time_limit: float = 1e60,
        verbose: bool = False,
    ) -> None:
        self.time_limit = time_limit
        self.relative_gap = relative_gap
        self.absolute_gap = absolute_gap
        self.verbose = verbose

        self.model = None

    def __str__(self):
        return "GurobiSolver"

    def build_model(
        self,
        datafit: MipDatafit,
        penalty: MipPenalty,
        A: NDArray,
        lmbd: float,
    ) -> gp.Model:

        self.m, self.n = A.shape
        self.datafit = datafit
        self.penalty = penalty
        self.A = A
        self.lmbd = lmbd

        self.model = gp.Model()

        self.model.setParam("OutputFlag", 1 if self.verbose else 0)
        self.model.setParam("TimeLimit", self.time_limit)
        self.model.setParam("MIPGap", self.relative_gap)
        self.model.setParam("MIPGapAbs", self.absolute_gap)

        self.x = self.model.addVars(
            self.n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x"
        )
        self.z = self.model.addVars(self.n, vtype=GRB.BINARY, name="z")
        self.w = self.model.addVars(
            self.m, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w"
        )
        self.f = self.model.addVar(
            lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="f"
        )
        self.h = self.model.addVar(
            lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="h"
        )

        for j in range(self.m):
            self.model.addConstr(
                self.w[j]
                == gp.quicksum(A[j, i] * self.x[i] for i in range(self.n))
            )

        self.bind_model(datafit)
        self.bind_model(penalty)

        self.model.setObjective(
            self.f
            + lmbd * gp.quicksum(self.z[i] for i in range(self.n))
            + self.h,
            GRB.MINIMIZE,
        )

        return self.model

    def bind_model(self, func: Union[BaseDatafit, BasePenalty]) -> None:
        if isinstance(func, Leastsquares):
            self.r = self.model.addVars(
                self.m, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS
            )
            self.model.addConstrs(
                self.r[j] == self.w[j] - func.y[j] for j in range(self.m)
            )
            self.model.addConstr(
                self.f
                >= 0.5
                * gp.quicksum(self.r[j] * self.r[j] for j in range(self.m))
            )
        elif isinstance(func, Logistic):
            self.r = self.model.addVars(
                self.m, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS
            )
            self.model.addConstrs(
                self.r[j] == -func.y[j] * self.w[j] for j in range(self.m)
            )
            self.model.addConstr(
                self.f
                == sum(
                    nlfunc.log(1.0 + nlfunc.exp(self.r[j]))
                    for j in range(self.m)
                )
            )
        elif isinstance(func, Squaredhinge):
            self.f1_var = self.model.addVars(
                self.m, lb=0.0, vtype=GRB.CONTINUOUS, name="f1"
            )
            self.model.addConstrs(
                self.f1_var[j] >= 1.0 - func.y[j] * self.w[j]
                for j in range(self.m)
            )
            self.model.addConstr(
                self.f
                >= gp.quicksum(
                    self.f1_var[j] * self.f1_var[j] for j in range(self.m)
                )
            )
        elif isinstance(func, Bigm):
            self.model.addConstrs(
                self.x[i] <= func.M * self.z[i] for i in range(self.n)
            )
            self.model.addConstrs(
                self.x[i] >= -func.M * self.z[i] for i in range(self.n)
            )
            self.model.addConstr(self.h >= 0.0)
        elif isinstance(func, BigmL1L2norm):
            self.h1_var = self.model.addVars(
                self.n, lb=0.0, ub=func.M, vtype=GRB.CONTINUOUS, name="h1"
            )
            self.h2_var = self.model.addVars(
                self.n, lb=0.0, ub=func.M**2, vtype=GRB.CONTINUOUS, name="h2"
            )
            self.model.addConstrs(
                self.x[i] <= func.M * self.z[i] for i in range(self.n)
            )
            self.model.addConstrs(
                self.x[i] >= -func.M * self.z[i] for i in range(self.n)
            )
            self.model.addConstrs(
                self.h1_var[i] >= self.x[i] for i in range(self.n)
            )
            self.model.addConstrs(
                self.h1_var[i] >= -self.x[i] for i in range(self.n)
            )
            for i in range(self.n):
                self.model.addQConstr(
                    self.x[i] * self.x[i] <= 2.0 * self.h2_var[i] * self.z[i]
                )
            self.model.addConstr(
                self.h
                >= func.alpha
                * gp.quicksum(self.h1_var[i] for i in range(self.n))
                + 2.0
                * func.beta
                * gp.quicksum(self.h2_var[i] for i in range(self.n))
            )
        elif isinstance(func, BigmL1norm):
            self.h_var = self.model.addVars(
                self.n, lb=0.0, ub=func.M, vtype=GRB.CONTINUOUS, name="h1"
            )
            self.model.addConstrs(
                self.x[i] <= func.M * self.z[i] for i in range(self.n)
            )
            self.model.addConstrs(
                self.x[i] >= -func.M * self.z[i] for i in range(self.n)
            )
            self.model.addConstrs(
                self.h_var[i] >= self.x[i] for i in range(self.n)
            )
            self.model.addConstrs(
                self.h_var[i] >= -self.x[i] for i in range(self.n)
            )
            self.model.addConstr(
                self.h
                >= func.alpha
                * gp.quicksum(self.h_var[i] for i in range(self.n))
            )
        elif isinstance(func, BigmL2norm):
            self.h_var = self.model.addVars(
                self.n, lb=0.0, ub=func.M**2, vtype=GRB.CONTINUOUS, name="h2"
            )
            self.model.addConstrs(
                self.x[i] <= func.M * self.z[i] for i in range(self.n)
            )
            self.model.addConstrs(
                self.x[i] >= -func.M * self.z[i] for i in range(self.n)
            )
            for i in range(self.n):
                self.model.addQConstr(
                    self.x[i] * self.x[i] <= 2.0 * self.h_var[i] * self.z[i]
                )
            self.model.addConstr(
                self.h
                >= 2.0
                * func.beta
                * gp.quicksum(self.h_var[i] for i in range(self.n))
            )
        elif isinstance(func, BigmPositiveL1norm):
            self.h_var = self.model.addVars(
                self.n, lb=0.0, ub=func.M, vtype=GRB.CONTINUOUS, name="h1"
            )
            self.model.addConstrs(
                self.x[i] <= func.M * self.z[i] for i in range(self.n)
            )
            self.model.addConstrs(self.x[i] >= 0.0 for i in range(self.n))
            self.model.addConstrs(
                self.h_var[i] >= self.x[i] for i in range(self.n)
            )
            self.model.addConstrs(self.h_var[i] >= 0.0 for i in range(self.n))
            self.model.addConstr(
                self.h
                >= func.alpha
                * gp.quicksum(self.h_var[i] for i in range(self.n))
            )
        elif isinstance(func, BigmPositiveL2norm):
            self.h_var = self.model.addVars(
                self.n, lb=0.0, ub=func.M**2, vtype=GRB.CONTINUOUS, name="h2"
            )
            self.model.addConstrs(
                self.x[i] <= func.M * self.z[i] for i in range(self.n)
            )
            self.model.addConstrs(self.x[i] >= 0.0 for i in range(self.n))
            for i in range(self.n):
                self.model.addQConstr(
                    self.x[i] * self.x[i] <= 2.0 * self.h_var[i] * self.z[i]
                )
            self.model.addConstr(
                self.h
                >= 2.0
                * func.beta
                * gp.quicksum(self.h_var[i] for i in range(self.n))
            )
        elif isinstance(func, Bounds):
            self.model.addConstrs(
                self.x[i] <= func.x_ub[i] * self.z[i] for i in range(self.n)
            )
            self.model.addConstrs(
                self.x[i] >= func.x_lb[i] * self.z[i] for i in range(self.n)
            )
            self.model.addConstr(self.h >= 0.0)
        elif isinstance(func, L1L2norm):
            self.h1_var = self.model.addVars(
                self.n, lb=0.0, vtype=GRB.CONTINUOUS, name="h1"
            )
            self.h2_var = self.model.addVars(
                self.n, lb=0.0, vtype=GRB.CONTINUOUS, name="h2"
            )
            self.model.addConstrs(
                self.h1_var[i] >= self.x[i] for i in range(self.n)
            )
            self.model.addConstrs(
                self.h1_var[i] >= -self.x[i] for i in range(self.n)
            )
            for i in range(self.n):
                self.model.addQConstr(
                    self.x[i] * self.x[i] <= 2.0 * self.h2_var[i] * self.z[i]
                )
            self.model.addConstr(
                self.h
                >= func.alpha
                * gp.quicksum(self.h1_var[i] for i in range(self.n))
                + 2.0
                * func.beta
                * gp.quicksum(self.h2_var[i] for i in range(self.n))
            )
        elif isinstance(func, L2norm):
            self.h_var = self.model.addVars(
                self.n, lb=0.0, vtype=GRB.CONTINUOUS, name="h2"
            )
            for i in range(self.n):
                self.model.addQConstr(
                    self.x[i] * self.x[i] <= 2.0 * self.h_var[i] * self.z[i]
                )
            self.model.addConstr(
                self.h
                >= 2.0
                * func.beta
                * gp.quicksum(self.h_var[i] for i in range(self.n))
            )
        elif isinstance(func, PositiveL2norm):
            self.h_var = self.model.addVars(
                self.n, lb=0.0, vtype=GRB.CONTINUOUS, name="h2"
            )
            self.model.addConstrs(self.x[i] >= 0.0 for i in range(self.n))
            for i in range(self.n):
                self.model.addQConstr(
                    self.x[i] * self.x[i] <= 2.0 * self.h_var[i] * self.z[i]
                )
            self.model.addConstr(
                self.h
                >= 2.0
                * func.beta
                * gp.quicksum(self.h_var[i] for i in range(self.n))
            )
        else:
            raise ValueError(f"Unsupported function type {type(func)}.")

    def package_result(self):
        if self.model.Status == GRB.OPTIMAL:
            status = Status.OPTIMAL
        elif self.model.Status == GRB.TIME_LIMIT:
            status = Status.TIME_LIMIT
        elif self.model.Status == GRB.UNBOUNDED:
            status = Status.UNBOUNDED
        elif self.model.Status == GRB.INFEASIBLE:
            status = Status.INFEASIBLE
        else:
            status = Status.UNKNOWN

        upper_bound = (
            float(self.model.ObjVal)
            if self.model.Status == GRB.OPTIMAL
            else np.inf
        )
        iter_count = int(self.model.IterCount)
        solve_time = float(self.model.Runtime)
        x = np.zeros(self.n)
        if self.model.Status == GRB.OPTIMAL:
            for i in range(self.n):
                x[i] = self.x[i].X * self.z[i].X

        return Result(
            status,
            solve_time,
            iter_count,
            x,
            upper_bound,
            None,
        )

    def solve(
        self,
        datafit: MipDatafit,
        penalty: MipPenalty,
        A: NDArray,
        lmbd: float,
        x_init: Optional[NDArray] = None,
    ):
        """Solve an L0-regularized problem.

        Parameters
        ----------
        datafit: MipDatafit
            Problem datafit function.
        penalty: MipDatafit
            Problem penalty function.
        A: NDArray
            Problem matrix.
        lmbd: float
            Problem L0-norm weight parameter.
        x_init: NDArray, default=None
            Initial point for the solver.
        """

        self.model = self.build_model(datafit, penalty, A, lmbd)

        if x_init is not None:
            assert len(x_init) == A.shape[1]
            for i, xi in enumerate(x_init):
                self.x[i].Start = xi

        self.model.optimize()

        return self.package_result()


class MosekSolver(MipSolver):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(optimizer_name="mosek", *args, **kwargs)


class CplexSolver(MipSolver):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(optimizer_name="cplex", *args, **kwargs)


def get_solver(type: str, args: dict) -> BaseSolver:
    if type == "el0ps":
        return El0psSolver(**args)
    elif type == "cplex":
        return CplexSolver(**args)
    elif type == "gurobi":
        return GurobiSolver(**args)
    elif type == "mosek":
        return MosekSolver(**args)
    elif type == "oa":
        return OaSolver(**args)
    elif type == "l0bnb":
        return L0bnbSolver(**args)
    elif type == "mimosa":
        return MimosaSolver(**args)
    else:
        raise ValueError(f"Unknown solver type {type}.")


def can_handle_instance(
    solver: BaseSolver, f: BaseDatafit, h: BasePenalty
) -> bool:
    if isinstance(solver, El0psSolver):
        return True
    elif isinstance(solver, MosekSolver):
        if not type(f) in _mip_supports["mosek"]["datafit"]:
            return False
        if not type(h) in _mip_supports["mosek"]["penalty"]:
            return False
        return True
    elif isinstance(solver, CplexSolver):
        if not type(f) in _mip_supports["cplex"]["datafit"]:
            return False
        if not type(h) in _mip_supports["cplex"]["penalty"]:
            return False
        return True
    elif isinstance(solver, OaSolver):
        return True
    elif isinstance(solver, L0bnbSolver):
        return type(f) in [Leastsquares] and type(h) in [
            Bigm,
            L2norm,
            BigmL2norm,
        ]
    elif isinstance(solver, MimosaSolver):
        return type(f) in [Leastsquares] and type(h) in [Bigm]
    elif isinstance(solver, GurobiSolver):
        return type(f) in [Leastsquares, Logistic, Squaredhinge] and type(
            h
        ) in [
            Bigm,
            BigmL1L2norm,
            BigmL1norm,
            BigmL2norm,
            BigmPositiveL1norm,
            BigmPositiveL2norm,
            Bounds,
            L1L2norm,
            L2norm,
            PositiveL2norm,
        ]
    else:
        raise ValueError(f"Unknown solver {solver}.")


def can_handle_compilation(solver: BaseSolver) -> bool:
    return isinstance(solver, (El0psSolver, OaSolver))


def get_result(
    solver: BaseSolver,
    f: BaseDatafit,
    h: BasePenalty,
    A: NDArray,
    l: float,
    type: str,
    args: Optional[dict],
) -> Result:

    if args is None:
        args = {}

    # Get compiled versions of datafit and penalty and warmup
    # compilation by running the solver with a short time limit
    if can_handle_compilation(solver):
        f = compiled_clone(f)
        h = compiled_clone(h)
        time_limit = solver.time_limit
        solver.time_limit = 5.0
        solver.solve(f, h, A, 0.1 * compute_lmbd_max(f, h, A))
        solver.time_limit = time_limit

    if type == "solve":
        result = solver.solve(f, h, A, l)
        print(result)
    elif type == "path":
        result = Path(**args).fit(solver, f, h, A)
    else:
        raise ValueError(f"Unknown action type {type}.")

    return result
