import os
import pathlib
import re
import shutil
import subprocess
import sys
import numpy as np
from typing import Union
from numpy.typing import ArrayLike
from el0ps.datafit import Leastsquares
from el0ps.penalty import Bigm, L2norm, BigmL2norm
from el0ps.solver import (
    BaseSolver,
    Status,
    Result,
    BnbSolver,
    MipSolver,
    OaSolver,
)
from l0bnb import BNBTree
from numba.experimental.jitclass.base import JitClassType


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
        A: ArrayLike,
        lmbd: float,
        x_init: Union[ArrayLike, None] = None,
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

        if 'MIMOSA_BIN' not in os.environ:
            raise ValueError("MIMOSA_BIN environment variable is not set.")
        self.MIMOSA_BIN = pathlib.Path(os.environ['MIMOSA_BIN']).absolute()

        if 'MIMOSA_TMP' not in os.environ:
            raise ValueError("MIMOSA_TMP environment variable is not set.")
        self.MIMOSA_TMP = pathlib.Path(os.environ['MIMOSA_TMP']).absolute()

    def __str__(self):
        return "MimosaSolver"

    def solve(
        self,
        datafit: Leastsquares,
        penalty: Bigm,
        A: ArrayLike,
        lmbd: float,
        x_init: Union[ArrayLike, None] = None,
    ) -> Result:

        assert isinstance(datafit, Leastsquares)
        assert isinstance(penalty, Bigm)

        # Clean and create MIMOSA_TMP
        if self.MIMOSA_TMP.is_dir():
            shutil.rmtree(self.MIMOSA_TMP)
        self.MIMOSA_TMP.mkdir()

        # Save instance to MIMOSA_TMP
        np.savetxt(self.MIMOSA_TMP / 'A.dat', A)
        np.savetxt(self.MIMOSA_TMP / 'y.dat', datafit.y)
        np.savetxt(self.MIMOSA_TMP / 'mu.dat', [lmbd])

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
        subprocess.run(command, shell=True)

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
            match = re.search(r"x_sol\s*((?:\s*-?\d*\.?\d+\s*)+)", content, re.DOTALL)  # noqa: E501
            if match:
                x_lines = match.group(1)
                x_vals = list(map(float, re.findall(r"-?\d*\.?\d+", x_lines)))
                x = np.array(x_vals)
            else:
                x = np.zeros(A.shape[1])
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


def get_solver(solver_name: str, solver_opts: dict) -> BaseSolver:
    if solver_name == "el0ps":
        return BnbSolver(**solver_opts)
    elif solver_name == "mip":
        return MipSolver(**solver_opts)
    elif solver_name == "oa":
        return OaSolver(**solver_opts)
    elif solver_name == "l0bnb":
        return L0bnbSolver(**solver_opts)
    elif solver_name == "mimosa":
        return MimosaSolver(**solver_opts)
    else:
        raise ValueError(f"Unknown solver {solver_name}.")


def can_handle_instance(
    solver_name: str,
    solver_opts: dict,
    datafit_name: str,
    penalty_name: str,
) -> bool:
    if solver_name == "el0ps":
        return True
    elif solver_name == "mip":
        optim_name = solver_opts["optimizer_name"]
        if optim_name == "cplex":
            return datafit_name in [
                "Leastsquares",
                "Squaredhinge",
            ] and penalty_name in [
                "Bigm",
                "BigmL1norm",
                "BigmL1L2norm",
                "BigmL2norm",
                "Bounds",
                "L2norm",
                "L1L2norm",
                "PositiveL2norm",
            ]
        elif optim_name == "gurobi":
            return datafit_name in [
                "Leastsquares",
                "Squaredhinge",
            ] and penalty_name in [
                "Bigm",
                "BigmL1norm",
                "BigmL1L2norm",
                "BigmL2norm",
                "Bounds",
                "L2norm",
                "L1L2norm",
                "PositiveL2norm",
            ]
        elif optim_name == "mosek":
            return datafit_name in [
                "Leastsquares",
                "Logistic",
                "Squaredhinge",
            ] and penalty_name in [
                "Bigm",
                "BigmL1norm",
                "BigmL1L2norm",
                "BigmL2norm",
                "Bounds",
                "L2norm",
                "L1L2norm",
                "PositiveL2norm",
            ]
        else:
            raise ValueError(f"Unknown optimizer {optim_name}.")
    elif solver_name == "oa":
        return True
    elif solver_name == "l0bnb":
        return datafit_name in ["Leastsquares"] and penalty_name in [
            "Bigm",
            "BigmL2norm",
            "L2norm",
        ]
    elif solver_name == "mimosa":
        return (datafit_name == "Leastsquares") and (penalty_name == "Bigm")
    else:
        raise ValueError(f"Unknown solver {solver_name}.")


def can_handle_compilation(solver_name: str) -> bool:
    return solver_name in ["el0ps", "oa"]


def precompile_solver(
    solver: BaseSolver,
    datafit: JitClassType,
    penalty: JitClassType,
    A: ArrayLike,
    lmbd: float,
    precompile_time: float = 5.0,
) -> None:

    time_limit = solver.time_limit
    solver.time_limit = precompile_time
    solver.solve(datafit, penalty, A, lmbd)
    solver.time_limit = time_limit
