import argparse
import pickle
import pathlib
import random
import string
import yaml
from datetime import datetime

from l0exp.dataset import get_dataset
from l0exp.calibration import get_calibration
from l0exp.solver import get_solver, get_result, can_handle_instance

from el0ps.utils import compute_lmbd_max

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()


# Load configuration
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
print("=" * 20 + f" Running experiment {config['experiment']} " + "=" * 20)


# Load dataset
print("Generating dataset...")
A, y, x = get_dataset(**config["dataset"])
print(f"  A shape: {A.shape}")
print(f"  y shape: {y.shape}")
print(f"  x shape: {x.shape if x is not None else 'None'}")
print()


# Calibrate problem datafit, penalty and l0-norm weight
print("Calibrating problem...")
f, h, lmbd = get_calibration(A, y, x, **config["calibration"])
print(f"  datafit: {f} {f.params_to_dict()}")
print(f"  penalty: {h} {h.params_to_dict()}")
print(f"  lambda : {lmbd} ({lmbd / compute_lmbd_max(f, h, A):.2e} * lmbd_max)")
print()


# Run each solver and collect results
results = {}
for solver_name, solver_config in config["solvers"].items():

    try:
        solver = get_solver(**solver_config)
    except Exception as e:
        print(f"Failed to initialize solver {solver_name} with error: {e}")
        print()
        results[solver_name] = None
        continue

    if can_handle_instance(solver, f, h):
        print(f"Running {solver_name}...")
        try:
            result = get_result(solver, f, h, A, lmbd, **config["action"])
        except Exception as e:
            print(f"  Solver failed with error: {e}")
            result = None
    else:
        print(f"Skipping {solver_name}")
        result = None
    print()

    results[solver_name] = result


# Save results
if args.save:
    name = config["experiment"]
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    uuid = "".join(random.choices(string.ascii_lowercase, k=10))
    file = f"{name}_{time}_{uuid}.pkl"
    path = pathlib.Path(__file__).parent / "results" / file
    data = {
        "config": config,
        "params": h.params_to_dict() | {"lambda": lmbd},
        "results": results,
    }

    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Results saved to {file}")
