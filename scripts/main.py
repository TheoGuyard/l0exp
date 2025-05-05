import argparse
import pathlib
import random
import shutil
import string
import subprocess
import yaml
from copy import deepcopy


# ----- Path variables ----- #

SRC_PATH = "~/Documents/Github/l0exp"
DST_PATH = "tguyard@cedar.alliancecan.ca:scratch"
HOME_DIR = "/home/tguyard"
VENV_DIR = "/home/tguyard/.venv"
LOGS_DIR = "/home/tguyard/logs"
BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
EXPS_DIR = BASE_DIR / "l0exp" / "experiments"
SCPT_DIR = BASE_DIR / "scripts"
TMPS_DIR = SCPT_DIR / "tmps"
RUN_PATH = SCPT_DIR / "run.sh"
JOB_NAME = "job.sh"


# ----- Experiments setups ----- #


def get_exp_mixtures():
    exp = {
        "name": "mixtures",
        "walltime": "11:00:00",
        "besteffort": True,
        "production": True,
        "setups": [],
        "repeats": 10,
    }

    time_limit = 600.0
    relative_gap = 1e-8
    verbose = False

    base_setup = {
        "expname": "mixtures",
        "dataset": {
            "k": 10,
            "m": 500,
            "n": 1000,
            "r": 0.9,
            "s": 10.0,
            "distrib_name": "gaussian",
            "distrib_opts": {"scale": 1.0},
        },
        "solvers": {
            "el0ps": {
                "solver": "el0ps",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "l0bnb": {
                "solver": "l0bnb",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "gurobi": {
                "solver": "mip",
                "params": {
                    "optimizer_name": "gurobi",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "oa": {
                "solver": "oa",
                "params": {
                    "optimizer_name": "gurobi",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
        },
    }

    for distrib_name, distrib_opts in [
        ("gaussian", {"scale": 1.0}),
        ("laplace", {"scale": 1.0}),
        ("uniform", {"low": 0.0, "high": 1.0}),
        ("halfgaussian", {"scale": 1.0}),
        ("halflaplace", {"scale": 1.0}),
        ("gausslaplace", {"scale1": 1.0, "scale2": 1.0}),
    ]:
        setup = deepcopy(base_setup)
        setup["dataset"]["distrib_name"] = distrib_name
        setup["dataset"]["distrib_opts"] = distrib_opts
        exp["setups"].append(setup)
    return exp


def get_exp_microscopy():
    exp = {
        "name": "microscopy",
        "walltime": "11:00:00",
        "besteffort": True,
        "production": True,
        "setups": [],
        "repeats": 10,
    }

    time_limit = 60.0
    relative_gap = 1e-8
    verbose = False

    base_setup = {
        "expname": "microscopy",
        "penalty": "BigmL1norm",
        "calibration": {"method": "l0learn", "kwargs": {}},
        "solvers": {
            "el0ps": {
                "solver": "el0ps",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "l0bnb": {
                "solver": "l0bnb",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "gurobi": {
                "solver": "mip",
                "params": {
                    "optimizer_name": "gurobi",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "oa": {
                "solver": "oa",
                "params": {
                    "optimizer_name": "gurobi",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
        },
        "path_opts": {
            "lmbd_max": 1e-0,
            "lmbd_min": 1e-3,
            "lmbd_num": 31,
            "lmbd_scaled": True,
            "stop_if_not_optimal": True,
            "verbose": True,
        },
    }

    for penalty in ["BigmL1norm", "BigmL2norm"]:
        setup = deepcopy(base_setup)
        setup["penalty"] = penalty
        exp["setups"].append(setup)

    return exp


def get_exp_realworld():
    exp = {
        "name": "realworld",
        "walltime": "01:00:00",
        "besteffort": False,
        "production": True,
        "setups": [],
        "repeats": 1,
    }

    time_limit = 60.0
    relative_gap = 1e-8
    verbose = False

    setup = {
        "expname": "realworld",
        "dataset": "riboflavin",
        "datafit": "Leastsquares",
        "penalty": "BigmL2norm",
        "calibration": {
            "method": "cv",
            "kwargs": {
                "criterion": "bic",
                "time_limit": 60.0,
            },
        },
        "solvers": {
            "el0ps": {
                "solver": "el0ps",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "l0bnb": {
                "solver": "l0bnb",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "gurobi": {
                "solver": "mip",
                "params": {
                    "optimizer_name": "gurobi",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "oa": {
                "solver": "oa",
                "params": {
                    "optimizer_name": "gurobi",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
        },
    }

    exp["setups"].append(setup)

    return exp


def get_exp_regpath():
    exp = {
        "name": "regpath",
        "walltime": "11:00:00",
        "besteffort": False,
        "production": True,
        "setups": [],
        "repeats": 1,
    }

    time_limit = 600.0
    relative_gap = 1e-8
    verbose = False

    base_setup = {
        "expname": "regpath",
        "dataset": "riboflavin",
        "datafit": "Leastsquares",
        "penalty": "BigmL2norm",
        "calibration": {"method": "l0learn", "kwargs": {}},
        "solvers": {
            "el0ps": {
                "solver": "el0ps",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "l0bnb": {
                "solver": "l0bnb",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "gurobi": {
                "solver": "mip",
                "params": {
                    "optimizer_name": "gurobi",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "oa": {
                "solver": "oa",
                "params": {
                    "optimizer_name": "gurobi",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
        },
        "path_opts": {
            "lmbd_max": 1e-0,
            "lmbd_min": 1e-3,
            "lmbd_num": 31,
            "lmbd_scaled": True,
            "stop_if_not_optimal": True,
            "verbose": True,
        },
    }

    for dataset, datafit in [
        ("riboflavin", "Leastsquares"),
        ("bctcga", "Leastsquares"),
        ("colon-cancer", "Logistic"),
        ("leukemia", "Logistic"),
        ("breast-cancer", "Squaredhinge"),
        ("arcene", "Squaredhinge"),
    ]:
        for penalty in ["BigmL1norm", "BigmL2norm"]:
            setup = deepcopy(base_setup)
            setup["dataset"] = dataset
            setup["datafit"] = datafit
            setup["penalty"] = penalty
            exp["setups"].append(setup)

    return exp


def get_exp_synthetic():
    exp = {
        "name": "synthetic",
        "walltime": "01:00:00",
        "besteffort": False,
        "production": True,
        "setups": [],
        "repeats": 1,
    }

    time_limit = 60.0
    relative_gap = 1e-8
    verbose = False

    setup = {
        "expname": "synthetic",
        "dataset": {
            "t": 0.0,
            "k": 2,
            "m": 30,
            "n": 50,
            "r": 0.1,
            "s": 10.0,
        },
        "penalty": "BigmL2norm",
        "calibration": {"method": "l0learn", "kwargs": {}},
        "solvers": {
            "el0ps": {
                "solver": "el0ps",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "l0bnb": {
                "solver": "l0bnb",
                "params": {
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "gurobi": {
                "solver": "mip",
                "params": {
                    "optimizer_name": "gurobi",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
            "oa": {
                "solver": "oa",
                "params": {
                    "optimizer_name": "gurobi",
                    "time_limit": time_limit,
                    "relative_gap": relative_gap,
                    "verbose": verbose,
                },
            },
        },
    }

    exp["setups"].append(setup)

    return exp


# ----- Experiments list ----- #

EXPERIMENTS = [
    get_exp_mixtures(),
    get_exp_microscopy(),
    get_exp_realworld(),
    get_exp_regpath(),
    get_exp_synthetic(),
]


# ----- System streams ----- #


def oar_run_steam():

    stream = "\n".join(
        [
            "#!/bin/sh",
            "expname=$1",
            "repeats=$2",
            "for i in $(seq 1 $repeats);",
            "do",
            "   oarsub --project simsmart -S {}/$expname/{}".format(
                TMPS_DIR,
                JOB_NAME,
            ),
            "done",
        ]
    )

    return stream


def oar_exp_steam(experiment, configs_path):
    stream = "\n".join(
        [
            "#!/bin/sh",
            "#OAR -n l0exp-{}".format(experiment["name"]),
            "#OAR -O {}/l0exp-{}.%jobid%.stdout".format(
                LOGS_DIR, experiment["name"]
            ),
            "#OAR -E {}/l0exp-{}.%jobid%.stderr".format(
                LOGS_DIR, experiment["name"]
            ),
            "#OAR -l walltime={}".format(experiment["walltime"]),
            "#OAR -t besteffort" if experiment["besteffort"] else "",
            "#OAR -q production" if experiment["production"] else "",
            "#OAR -p gpu_count=0",
            "#OAR --array-param-file {}".format(configs_path),
            "set -xv",
            "source {}/.profile".format(HOME_DIR),
            "module load conda gurobi cplex",
            "conda activate l0exp",
            "{} {}/{}/exp.py run -r {}/{}/results -c $* -n {} -v".format(
                "python",
                EXPS_DIR,
                experiment["name"],
                EXPS_DIR,
                experiment["name"],
                experiment["repeats"],
            ),
        ]
    )

    return stream


def slurm_run_steam():

    stream = "\n".join(
        [
            "#!/bin/sh",
            "expname=$1",
            "repeats=$2",
            "for i in $(seq 1 $repeats);",
            "do",
            "   sbatch {}/$expname/{}".format(TMPS_DIR, JOB_NAME),
            "done",
        ]
    )

    return stream


def slurm_exp_steam(experiment, configs_path):

    with open(configs_path, "r") as fp:
        num_configs = len(fp.readlines())

    stream = "\n".join(
        [
            "#!/bin/sh",
            "#SBATCH -J l0exp-{}".format(experiment["name"]),
            "#SBATCH -o {}/%x.%j.out".format(LOGS_DIR),
            "#SBATCH -e {}/%x.%j.err".format(LOGS_DIR),
            "#SBATCH -t {}".format(experiment["walltime"]),
            "#SBATCH --array=0-{}".format(num_configs - 1),
            "#SBATCH --account=def-vidalthi",
            'cp=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {})'.format(
                configs_path
            ),
            "set -xv",
            "source {}/.bash_profile".format(HOME_DIR),
            "module load python mpi4py gurobi",
            "source {}/.venv/bin/activate".format(HOME_DIR),
            "{} {}/{}/exp.py run -r {}/{}/results -c $cp -n {} -v".format(
                "python",
                EXPS_DIR,
                experiment["name"],
                EXPS_DIR,
                experiment["name"],
                experiment["repeats"],
            ),
        ]
    )

    return stream


# ----- Scripts functions ----- #


def send():
    print("send")
    cmd_str = " ".join(
        [
            "rsync -amv",
            "--exclude '.git'",
            "--exclude '.github'",
            "--exclude '.venv'",
            "--exclude '.DS_Store'",
            "--exclude 'doc/'",
            "--exclude 'build/'",
            "--exclude '**/results/*.pkl'",
            "--exclude '**/saves/*.csv'",
            "--exclude '**/saves/*.pkl'",
            "--exclude '**/__pycache__'",
            "--exclude '**/.pytest_cache'",
            "{} {}".format(SRC_PATH, DST_PATH),
        ]
    )
    subprocess.run(cmd_str, shell=True)


def install():
    print("install")
    cmd_strs = [
        "module load python mpi4py",
        "{}/bin/activate".format(VENV_DIR),
        "pip install -e {}".format(BASE_DIR),
    ]
    for cmd_str in cmd_strs:
        subprocess.run(cmd_str, shell=True)


def make(system):

    print("make run")

    # Run file stream
    if system == "oar":
        stream = oar_run_steam()
    elif system == "slurm":
        stream = slurm_run_steam()
    else:
        raise ValueError("Unknown system: {}".format(system))

    # Write run file
    with open(RUN_PATH, "w") as file:
        file.write(stream)
    subprocess.run("chmod u+x {}".format(RUN_PATH), shell=True)

    # Create the scripts dir (remove old one)
    if TMPS_DIR.is_dir():
        shutil.rmtree(TMPS_DIR)
    else:
        TMPS_DIR.mkdir()

    for experiment in EXPERIMENTS:
        print("make {}".format(experiment["name"]))

        # Create the experiment dir
        experiment_dir = TMPS_DIR.joinpath(experiment["name"])
        experiment_dir.mkdir()

        # Create the args file (remove old ones)
        configs_path = experiment_dir.joinpath("configs.txt")
        if configs_path.is_file():
            configs_path.unlink()

        # Create setups dir
        setups_dir = experiment_dir.joinpath("setups")
        setups_dir.mkdir()

        # Create setups and configs file
        for setup in experiment["setups"]:
            setup_name = "".join(
                random.choice(string.ascii_lowercase) for _ in range(10)
            )
            setup_file = "{}.yml".format(setup_name)
            setup_path = pathlib.Path(setups_dir, setup_file)

            with open(setup_path, "w") as file:
                yaml.dump(setup, file)

            with open(configs_path, "a") as file:
                file.write(str(setup_path))
                file.write("\n")

        # Oar file stream
        if system == "oar":
            stream = oar_exp_steam(experiment, configs_path)
        elif system == "slurm":
            stream = slurm_exp_steam(experiment, configs_path)
        else:
            raise ValueError("Unknown system: {}".format(system))

        # Write job file
        job_path = experiment_dir.joinpath(JOB_NAME)
        with open(job_path, "w") as file:
            file.write(stream)
        subprocess.run("chmod u+x {}".format(job_path), shell=True)


def receive(expname):
    for experiment in EXPERIMENTS:
        if (expname is None) or experiment["name"] == expname:
            print(f"receive {experiment['name']}")
            results_src_path = pathlib.Path(
                DST_PATH,
                "l0exp",
                "l0exp",
                "experiments",
                experiment["name"],
                "results/*",
            )
            results_dst_path = pathlib.Path(
                EXPS_DIR, experiment["name"], "results"
            )
            cmd_str = "rsync -amv {} {}".format(
                results_src_path, results_dst_path
            )
            subprocess.run(cmd_str, shell=True)


def clean():
    print("clean")
    if RUN_PATH.is_file():
        RUN_PATH.unlink()
    if TMPS_DIR.is_dir():
        shutil.rmtree(TMPS_DIR)


# ----- Command line interface ----- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cmd", choices=["send", "install", "make", "receive", "clean"]
    )
    parser.add_argument("-s", "--system", default="slurm")
    parser.add_argument("-n", "--expname", default=None)
    args = parser.parse_args()

    if args.cmd == "send":
        send()
    elif args.cmd == "install":
        install()
    elif args.cmd == "make":
        make(args.system)
    elif args.cmd == "clean":
        clean()
    elif args.cmd == "receive":
        receive(args.expname)
    else:
        raise ValueError(f"Unknown command {args.oar_cmd}.")
