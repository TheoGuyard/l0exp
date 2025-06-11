import argparse
import pathlib
from exprun import Runner


def get_experiment_from_str(expname: str) -> type:
    """
    Returns the experiment class based on the name.
    """
    if expname == "calibration":
        from l0exp.experiments.calibration.exp import Calibration

        return Calibration
    elif expname == "mixtures":
        from l0exp.experiments.mixtures.exp import Mixtures

        return Mixtures
    elif expname == "realworld":
        from l0exp.experiments.realworld.exp import Realworld

        return Realworld
    elif expname == "regpath":
        from l0exp.experiments.regpath.exp import Regpath

        return Regpath
    elif expname == "synthetic":
        from l0exp.experiments.synthetic.exp import Synthetic

        return Synthetic
    else:
        raise ValueError(f"Unknown experiment name: {expname}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "expname",
        type=str,
        help="Name of the experiment to run.",
    )
    parser.add_argument(
        "command",
        type=str,
        choices=["run", "plot"],
        help="Command to execute.",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=pathlib.Path,
        default=None,
        help="Path to the configuration file. If None, the default "
        "configuration file in the experiment's folder is used.",
    )
    parser.add_argument(
        "--result_dir",
        "-r",
        type=pathlib.Path,
        default=None,
        help="Path to the configuration result directory. If None, the"
        "default result directory in the experiment's folder is used.",
    )
    parser.add_argument(
        "--save_dir",
        "-s",
        type=pathlib.Path,
        default=None,
        help="Path to the configuration save directory. If None, the default"
        "save directory in the experiment's folder is used.",
    )
    parser.add_argument(
        "--repeats",
        "-n",
        type=int,
        default=1,
        help="Number of times to repeat the experiment. Default is 1.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to display information during the experiment run.",
    )
    args = parser.parse_args()

    runner = Runner(verbose=args.verbose)
    experiment = get_experiment_from_str(args.expname)

    experiments_dir = pathlib.Path(__file__).parent / "experiments"
    experiment_dir = experiments_dir / args.expname

    if args.config_path is None:
        args.config_path = experiment_dir / "exp.yml"
    if args.result_dir is None:
        args.result_dir = experiment_dir / "results"
    if args.save_dir is None:
        args.save_dir = experiment_dir / "saves"

    if args.command == "run":
        runner.run(
            experiment,
            args.config_path,
            args.result_dir,
            args.repeats,
        )
    elif args.command == "plot":
        runner.plot(
            experiment,
            args.config_path,
            args.result_dir,
            args.save_dir,
        )
    else:
        raise ValueError(f"Unknown command {args.command}.")
