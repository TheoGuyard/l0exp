from exprun import Experiment
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
            "lmbd": lmbd,
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
