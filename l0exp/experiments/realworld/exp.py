import argparse
import pathlib
from exprun import Experiment, Runner
from el0ps.compilation import CompilableClass, compiled_clone

from l0exp.experiments.dataset import load_dataset
from l0exp.experiments.solver import (
    get_solver,
    can_handle_instance,
    can_handle_compilation,
    precompile_solver,
)
from l0exp.experiments.instance import calibrate_parameters, preprocess_data


class Realworld(Experiment):

    def setup(self) -> None:

        A, y = load_dataset(self.config["dataset"])

        A, y, _ = preprocess_data(
            A,
            y,
            None,
            center=True,
            normalize=True,
            y_binary=self.config["datafit"] in ["Logistic", "Squaredhinge"],
        )

        datafit, penalty, lmbd, x_cal = calibrate_parameters(
            self.config["calibration"]["method"],
            self.config["datafit"],
            self.config["penalty"],
            A,
            y,
            **(
                self.config["calibration"]["kwargs"]
                if self.config["calibration"]["kwargs"] is not None
                else {}
            ),
        )

        self.x_cal = x_cal
        self.datafit = datafit
        self.penalty = penalty
        self.A = A
        self.lmbd = lmbd

        if isinstance(self.datafit, CompilableClass):
            self.datafit_compiled = compiled_clone(self.datafit)
        else:
            self.datafit_compiled = None
        if isinstance(self.penalty, CompilableClass):
            self.penalty_compiled = compiled_clone(self.penalty)
        else:
            self.penalty_compiled = None

    def run(self) -> dict:
        result = {}
        for solver_name, solver_keys in self.config["solvers"].items():
            if can_handle_instance(
                solver_keys["solver"],
                solver_keys["params"],
                str(self.datafit),
                str(self.penalty),
            ):
                print("Running {}...".format(solver_name))
                solver = get_solver(
                    solver_keys["solver"],
                    solver_keys["params"],
                )
                if can_handle_compilation(solver_keys["solver"]):
                    precompile_solver(
                        solver,
                        self.datafit_compiled,
                        self.penalty_compiled,
                        self.A,
                        self.lmbd,
                    )
                    result[solver_name] = solver.solve(
                        self.datafit_compiled,
                        self.penalty_compiled,
                        self.A,
                        self.lmbd,
                    )
                else:
                    result[solver_name] = solver.solve(
                        self.datafit, self.penalty, self.A, self.lmbd
                    )
            else:
                print("Skipping {}".format(solver_name))
                result[solver_name] = None

            if result[solver_name] is not None:
                print(result[solver_name])

        return result

    def cleanup(self) -> None:
        pass

    def plot(self, results: list) -> None:
        for result in results:
            for solver_name, solver_result in result.items():
                print(solver_name)
                print(solver_result)
                print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=["run", "plot"])
    parser.add_argument("--config_path", "-c", type=pathlib.Path)
    parser.add_argument("--results_dir", "-r", type=pathlib.Path)
    parser.add_argument("--repeats", "-n", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    runner = Runner(verbose=args.verbose)

    if args.command == "run":
        runner.run(Realworld, args.config_path, args.results_dir, args.repeats)
    elif args.command == "plot":
        runner.plot(Realworld, args.config_path, args.results_dir)
    else:
        raise ValueError(f"Unknown command {args.command}.")
