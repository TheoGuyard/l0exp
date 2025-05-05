import argparse
import pathlib
from exprun import Experiment, Runner
from el0ps.datafit import *  # noqa
from el0ps.penalty import *  # noqa

from l0exp.experiments.dataset import load_dataset
from l0exp.experiments.instance import calibrate_parameters


class Calibration(Experiment):

    def setup(self) -> None:

        A, y = load_dataset(self.config["dataset"])

        self.A = A
        self.y = y

    def run(self) -> dict:

        datafit, penalty, lmbd, x_cal = calibrate_parameters(
            self.config["calibration"]["method"],
            self.config["datafit"],
            self.config["penalty"],
            self.A,
            self.y,
            **(
                self.config["calibration"]["kwargs"]
                if self.config["calibration"]["kwargs"] is not None
                else {}
            ),
        )

        results = {
            "penalty_params": penalty.params_to_dict(),
            "lambda": lmbd,
            "x_cal": x_cal,
        }

        return results

    def cleanup(self) -> None:
        pass

    def plot(self, results: list) -> None:

        for result in results:
            for k, v in result.items():
                print(f"{k}: {v}")

    def save_plot(self, table, save_dir):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=["run", "plot"])
    parser.add_argument("--config_path", "-c", type=pathlib.Path)
    parser.add_argument("--result_dir", "-r", type=pathlib.Path)
    parser.add_argument("--save_dir", "-s", type=pathlib.Path, default=None)
    parser.add_argument("--repeats", "-n", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    runner = Runner(verbose=args.verbose)

    if args.command == "run":
        runner.run(
            Calibration,
            args.config_path,
            args.result_dir,
            args.repeats,
        )
    elif args.command == "plot":
        runner.plot(
            Calibration,
            args.config_path,
            args.result_dir,
            args.save_dir,
        )
    else:
        raise ValueError(f"Unknown command {args.command}.")
