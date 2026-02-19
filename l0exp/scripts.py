import argparse
import ast
import csv
import pathlib
import shutil
import subprocess
import textwrap
import yaml
from copy import deepcopy

# ----- Global variables ----- #

LOC_PATH = "~/Documents/Github/l0exp"
REM_PATH = "tguyard@narval.alliancecan.ca:scratch"
HOME_DIR = "/home/tguyard"
VENV_DIR = "/home/tguyard/.venv"
LOGS_DIR = "/home/tguyard/logs"
BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
EXPS_DIR = BASE_DIR / "l0exp"
TMPS_DIR = EXPS_DIR / "tmps"
RUN_PATH = EXPS_DIR / "run.sh"
JOB_NAME = "job.sh"
MODULES = ["arrow", "gcc", "gurobi", "mpi4py", "r"]


# ----- Job definitions helpers ----- #


def get_calibration_from_file(path, dataset, datafit, penalty):
    with open(path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if (
                row["dataset"] == dataset
                and row["datafit"] == datafit
                and row["penalty"] == penalty
            ):
                calibration = {
                    "datafit": datafit,
                    "penalty": penalty,
                    "lmbd": float(row["lmbd"]),
                    "datafit_params": ast.literal_eval(row["datafit_params"]),
                    "penalty_params": ast.literal_eval(row["penalty_params"]),
                }
                return calibration

    raise ValueError(
        "Calibration not found for {}, {}, {}.".format(
            dataset, datafit, penalty
        )
    )


# ----- Job definitions ----- #


EXPERIMENTS = [
    # ----- Section 6.2 calibration ----- #
    {
        "name": "calibration",
        "walltime": "04:00:00",
        "memory": "16G",
        "configs": [
            {
                "experiment": "calibration",
                "dataset": {
                    "type": "hardcoded",
                    "args": {"name": dataset, "normalize": True},
                },
                "calibration": {
                    "type": "cv",
                    "args": {"datafit": datafit, "penalty": penalty},
                },
                "solvers": {},
                "action": {},
            }
            for (dataset, datafit, penalty) in [
                ("drug", "Leastsquares", "BigmL1L2norm"),
                ("drug", "Leastsquares", "BigmL2norm"),
                #
                ("kits", "Leastsquares", "BigmL1L2norm"),
                ("kits", "Leastsquares", "BigmL2norm"),
                #
                ("gisette", "Logistic", "BigmL1L2norm"),
                ("gisette", "Logistic", "BigmL2norm"),
                ("gisette", "Squaredhinge", "BigmL1L2norm"),
                ("gisette", "Squaredhinge", "BigmL2norm"),
                #
                ("dorothea", "Logistic", "BigmL1L2norm"),
                ("dorothea", "Logistic", "BigmL2norm"),
                ("dorothea", "Squaredhinge", "BigmL1L2norm"),
                ("dorothea", "Squaredhinge", "BigmL2norm"),
                #
                ("madelon", "Logistic", "BigmL1L2norm"),
                ("madelon", "Logistic", "BigmL2norm"),
                ("madelon", "Squaredhinge", "BigmL1L2norm"),
                ("madelon", "Squaredhinge", "BigmL2norm"),
                #
                ("dexter", "Logistic", "BigmL1L2norm"),
                ("dexter", "Logistic", "BigmL2norm"),
                ("dexter", "Squaredhinge", "BigmL1L2norm"),
                ("dexter", "Squaredhinge", "BigmL2norm"),
            ]
        ],
    },
    # ----- Section 6.2 experiment ----- #
    {
        "name": "section_6.2",
        "walltime": "24:00:00",
        "memory": "256G",
        "configs": [
            {
                "experiment": "section_6.2",
                "dataset": {
                    "type": "hardcoded",
                    "args": {"name": dataset, "normalize": True},
                },
                "calibration": {
                    "type": "hardcoded",
                    "args": get_calibration_from_file(
                        pathlib.Path(__file__).parent
                        / "configs"
                        / "mpc_2025"
                        / "section_6.2_calibration.csv",
                        dataset,
                        datafit,
                        penalty,
                    ),
                },
                "solvers": {
                    solver: {
                        "type": solver,
                        "args": {
                            "time_limit": 3600.0,
                            "relative_gap": 1.0e-8,
                            "verbose": False,
                        },
                    }
                    for solver in solvers
                },
                "action": {
                    "type": "path",
                    "args": {"lmbd_max": 1, "lmbd_min": 0.01, "lmbd_num": 21},
                },
            }
            for (dataset, datafit, penalty, solvers) in [
                (
                    "drug",
                    "Leastsquares",
                    "BigmL1L2norm",
                    ["el0ps", "l0bnb", "gurobi", "mosek", "oa"],
                ),
                ("drug", "Leastsquares", "BigmL2norm", ["el0ps", "l0bnb", "gurobi", "mosek", "oa"]),
                ("kits", "Leastsquares", "BigmL1L2norm", ["el0ps", "l0bnb", "gurobi", "mosek", "oa"]),
                ("kits", "Leastsquares", "BigmL2norm", ["oa"]),
                ("gisette", "Logistic", "BigmL1L2norm", ["el0ps", "gurobi", "mosek", "oa"]),
                ("gisette", "Logistic", "BigmL2norm", ["el0ps", "gurobi", "mosek", "oa"]),
                ("madelon", "Logistic", "BigmL1L2norm", ["el0ps", "gurobi", "mosek", "oa"]),
                ("madelon", "Logistic", "BigmL2norm", ["el0ps", "gurobi", "mosek", "oa"]),
                (
                    "dorothea",
                    "Squaredhinge",
                    "BigmL1L2norm",
                    ["el0ps", "gurobi", "mosek", "oa"],
                ),
                (
                    "dorothea",
                    "Squaredhinge",
                    "BigmL2norm",
                    ["el0ps", "gurobi", "mosek", "oa"],
                ),
                ("dexter", "Squaredhinge", "BigmL1L2norm", ["el0ps", "gurobi", "mosek", "oa"]),
                ("dexter", "Squaredhinge", "BigmL2norm", ["el0ps", "gurobi", "mosek", "oa"]),
            ]
        ],
    },
    # ----- Section 6.3 experiment ----- #
    {
        "name": "section_6.3",
        "walltime": "01:15:00",
        "memory": "16G",
        "configs": [
            {
                "experiment": "section_6.3",
                "dataset": {
                    "type": "mixture",
                    "args": {
                        "k": 10,
                        "m": 500,
                        "n": 1000,
                        "r": 0.9,
                        "s": 10.0,
                        "distrib_name": distrib_name,
                        "distrib_args": deepcopy(distrib_args),
                    },
                },
                "calibration": {
                    "type": "mixture",
                    "args": {
                        "distrib_name": distrib_name,
                        "distrib_args": deepcopy(distrib_args),
                    },
                },
                "solvers": {
                    solver: {
                        "type": solver,
                        "args": {
                            "time_limit": 600.0,
                            "relative_gap": 1.0e-8,
                            "verbose": False,
                        },
                    }
                    for solver in [
                        "el0ps",
                        "l0bnb",
                        "gurobi",
                        "mosek",
                        "oa",
                    ]
                },
                "action": {"type": "solve", "args": {}},
            }
            for (distrib_name, distrib_args) in [
                ("uniform", {"low": -1.0, "high": 1.0}),
                ("gaussian", {"scale": 1.0}),
                ("laplace", {"scale": 1.0}),
                ("exponential", {"scale": 1.0}),
                ("halfgaussian", {"scale": 1.0}),
                ("gausslaplace", {"scale1": 1.0, "scale2": 1.0}),
            ]
        ],
    },
    # ----- Section 6.4 experiment ----- #
    {
        "name": "section_6.4",
        "walltime": "05:30:00",
        "memory": "16G",
        "configs": [
            {
                "experiment": "section_6.4",
                "dataset": {
                    "type": "mixture",
                    "args": {
                        "k": k,
                        "m": m,
                        "n": n,
                        "r": r,
                        "s": s,
                        "distrib_name": "uniform",
                        "distrib_args": {"low": -bigm, "high": bigm},
                    },
                },
                "calibration": {
                    "type": "mixture",
                    "args": {
                        "distrib_name": "uniform",
                        "distrib_args": {"low": -bigm, "high": bigm},
                    },
                },
                "solvers": {
                    solver: {
                        "type": solver,
                        "args": {
                            "time_limit": 3600.0,
                            "relative_gap": 1.0e-8,
                            "verbose": False,
                        },
                    }
                    for solver in [
                        "el0ps",
                        "l0bnb",
                        "gurobi",
                        "mosek",
                        "oa",
                    ]
                },
                "action": {"type": "solve", "args": {}},
            }
            for (k, m, n, r, s, bigm) in [
                # # Vary k
                # (5, 500, 1000, 0.9, 10., 1.0),
                # (7, 500, 1000, 0.9, 10., 1.0),
                # (9, 500, 1000, 0.9, 10., 1.0),
                # (11, 500, 1000, 0.9, 10., 1.0),
                # (13, 500, 1000, 0.9, 10., 1.0),
                # (15, 500, 1000, 0.9, 10., 1.0),
                # # Vary m
                # (10, 100, 1000, 0.9, 10., 1.0),
                # (10, 316, 1000, 0.9, 10., 1.0),
                # (10, 1000, 1000, 0.9, 10., 1.0),
                # (10, 3162, 1000, 0.9, 10., 1.0),
                # (10, 10000, 1000, 0.9, 10., 1.0),
                # # Vary n
                # (10, 500, 100, 0.9, 10., 1.0),
                # (10, 500, 316, 0.9, 10., 1.0),
                # (10, 500, 1000, 0.9, 10., 1.0),
                # (10, 500, 3162, 0.9, 10., 1.0),
                # (10, 500, 10000, 0.9, 10., 1.0),
                # # Vary r
                # (10, 500, 1000, 0.0, 10., 1.0),
                # (10, 500, 1000, 0.68377223, 10., 1.0),
                # (10, 500, 1000, 0.9, 10., 1.0),
                # (10, 500, 1000, 0.96837722, 10., 1.0),
                # (10, 500, 1000, 0.99, 10., 1.0),
                # # Vary s
                # (10, 500, 1000, 0.9, 1., 1.0),
                # (10, 500, 1000, 0.9, 3.16227766, 1.0),
                # (10, 500, 1000, 0.9, 10., 1.0),
                # (10, 500, 1000, 0.9, 31.6227766, 1.0),
                # (10, 500, 1000, 0.9, 100., 1.0),
                # Vary bigm factor
                (10, 500, 1000, 0.9, 10.0, 1.0),
                (10, 500, 1000, 0.9, 10.0, 2.0),
                (10, 500, 1000, 0.9, 10.0, 4.0),
                (10, 500, 1000, 0.9, 10.0, 6.0),
                (10, 500, 1000, 0.9, 10.0, 8.0),
                (10, 500, 1000, 0.9, 10.0, 10.0),
            ]
        ],
    },
]


# ----- Slurm bash files ----- #


def slurm_run_stream():
    steam = textwrap.dedent("""\
        #!/bin/sh
        expname=$1
        repeats=$2
        for i in $(seq 1 $repeats);
        do
            sbatch {}/$expname/{}
        done
    """.format(TMPS_DIR, JOB_NAME))
    return steam


def slurm_job_stream(experiment: dict):
    stream = textwrap.dedent(
        """\
        #!/bin/sh
        #SBATCH -J {}
        #SBATCH -o {}/%x_%A_%a.out
        #SBATCH -e {}/%x_%A_%a.err
        {}
        {}
        {}
        #SBATCH --array=0-{}
        #SBATCH --account=def-vidalthi

        config_file="{}_${{SLURM_ARRAY_TASK_ID}}.yml"

        module load {}
        source {}/.bash_profile
        source {}/.bashrc
        source {}/.venv/bin/activate
        cd {}
        python ./l0exp/run.py --config {}/{}/configs/$config_file --save
    """.format(
            experiment["name"],
            LOGS_DIR,
            LOGS_DIR,
            (
                "#SBATCH -t {}".format(experiment["walltime"])
                if "walltime" in experiment
                else ""
            ),
            (
                "#SBATCH --mem={}".format(experiment["memory"])
                if "memory" in experiment
                else ""
            ),
            (
                "#SBATCH --cpus-per-task={}".format(experiment["cpus"])
                if "cpus" in experiment
                else ""
            ),
            len(experiment["configs"]) - 1,
            experiment["name"],
            " ".join(MODULES),
            HOME_DIR,
            HOME_DIR,
            HOME_DIR,
            BASE_DIR,
            TMPS_DIR,
            experiment["name"],
        )
    )
    return stream


# ----- Command line functions ----- #


def send():
    print("send")
    cmd_str = " ".join(
        [
            "rsync -amv",
            "--exclude '.git'",
            "--exclude '.github'",
            "--exclude '.venv'",
            "--exclude '.DS_Store'",
            "--exclude 'reproduce'",
            "--exclude '**/results/*.pkl'",
            "--exclude '**/saves/*.csv'",
            "--exclude '**/__pycache__'",
            "--exclude '**/.pytest_cache'",
            "{} {}".format(LOC_PATH, REM_PATH),
        ]
    )
    subprocess.run(cmd_str, shell=True)


def install():
    print("install")
    cmd_str = textwrap.dedent(
        """\
        module load {}
        source {}/.bash_profile
        source {}/.bashrc
        source {}/.venv/bin/activate
        cd {}
        pip install -e .
    """.format(
            " ".join(MODULES),
            HOME_DIR,
            HOME_DIR,
            HOME_DIR,
            BASE_DIR,
        )
    )
    subprocess.run(cmd_str, shell=True)


def make():

    # Run file stream
    run_stream = slurm_run_stream()

    # Write run file
    with open(RUN_PATH, "w") as file:
        file.write(run_stream)
    subprocess.run("chmod u+x {}".format(RUN_PATH), shell=True)

    # Create temporary dir for each job array (and remove old one if needed)
    if TMPS_DIR.is_dir():
        shutil.rmtree(TMPS_DIR)
    TMPS_DIR.mkdir()

    for experiment in EXPERIMENTS:
        print("make {}".format(experiment["name"]))

        # Create the job array dir
        job_array_dir = TMPS_DIR.joinpath(experiment["name"])
        if job_array_dir.is_dir():
            shutil.rmtree(job_array_dir)
        job_array_dir.mkdir()

        # Create the job array configs dir
        job_array_configs_dir = job_array_dir.joinpath("configs")
        if job_array_configs_dir.is_dir():
            shutil.rmtree(job_array_configs_dir)
        job_array_configs_dir.mkdir()

        # Write the configs file
        for i, config in enumerate(experiment["configs"]):
            with open(
                job_array_configs_dir.joinpath(
                    f"{experiment['name']}_{i:d}.yml"
                ),
                "w",
            ) as file:
                yaml.dump(config, file)

        # Slurm job stream
        job_stream = slurm_job_stream(experiment)

        # Write job file
        job_path = job_array_dir.joinpath(JOB_NAME)
        with open(job_path, "w") as file:
            file.write(job_stream)
        subprocess.run("chmod u+x {}".format(job_path), shell=True)


def receive(filefilter="*.pkl"):
    print("receive")
    src_path = pathlib.Path(REM_PATH, "l0exp", "l0exp", "results", filefilter)
    dst_path = pathlib.Path(LOC_PATH, "l0exp", "results")
    cmd_str = "rsync -amv {} {}".format(src_path, dst_path)
    subprocess.run(cmd_str, shell=True)


def clean():
    print("clean")
    if RUN_PATH.is_file():
        RUN_PATH.unlink()
    if TMPS_DIR.is_dir():
        shutil.rmtree(TMPS_DIR)


def run(expname, repeats=1):
    subprocess.run("{} {} {}".format(RUN_PATH, expname, repeats), shell=True)


# ----- Command line interface ----- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cmd", choices=["send", "install", "make", "run", "clean", "receive"]
    )
    parser.add_argument("-e", "--expname", default=None)
    parser.add_argument("-r", "--repeats", type=int, default=1)
    parser.add_argument("-f", "--filefilter", type=str, default="*.pkl")
    args = parser.parse_args()

    if args.cmd == "send":
        send()
    elif args.cmd == "install":
        install()
    elif args.cmd == "make":
        make()
    elif args.cmd == "clean":
        clean()
    elif args.cmd == "receive":
        receive(args.filefilter)
    elif args.cmd == "run":
        if args.expname is None:
            raise ValueError("Experiment name must be provided for receive.")
        run(args.expname, args.repeats)
    else:
        raise ValueError(f"Unknown command {args.cmd}.")
