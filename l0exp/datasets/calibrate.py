import argparse
import csv
import pathlib
import time
from l0exp.calibration import get_calibration_cv
from l0exp.dataset import get_dataset_hardcoded, get_dataset_realworld

OUTPUT_PATH = pathlib.Path(__file__).parent / f"calibrate.csv"
FIELDS = [
    "dataset",
    "m",
    "n",
    "datafit",
    "penalty",
    "method",
    "nfolds",
    "factor",
    "lmbd",
    "datafit_params",
    "penalty_params",
]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--datafit", type=str, required=True, help="Datafit function to use."
    )
    parser.add_argument(
        "--penalty", type=str, required=True, help="Penalty function to use."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cvx",
        choices=["omp", "iht", "cvx"],
        help="Method to use for calibration.",
    )
    parser.add_argument(
        "--nfolds",
        type=int,
        default=10,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=1.5,
        help="Factor to multiply the big-M constant by.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to print verbose output.",
    )

    args = parser.parse_args()

    print(f"Loading dataset...")
    try:
        A, y, x = get_dataset_hardcoded(args.dataset)
    except FileNotFoundError:
        print("Dataset not found under l0exp/datasets. Fetching it...")
        try:
            A, y, x = get_dataset_realworld(args.dataset)
        except:
            raise ValueError(f"Fetching of dataset '{args.dataset}' failed.")

    print("Running calibration...")
    print("  datatset:", args.dataset)
    print("  shape   :", A.shape)
    print("  datafit :", args.datafit)
    print("  penalty :", args.penalty)
    print("  method  :", args.method)
    print("  nfolds  :", args.nfolds)
    print("  factor  :", args.factor)
    start = time.time()
    f, h, lmbd = get_calibration_cv(
        A,
        y,
        x,
        args.datafit,
        args.penalty,
        method=args.method,
        nfolds=args.nfolds,
        factor=args.factor,
        verbose=args.verbose,
    )
    print(f"  done in {time.time() - start:.2f}s")

    row = {
        "dataset": args.dataset,
        "m": A.shape[0],
        "n": A.shape[1],
        "datafit": args.datafit,
        "penalty": args.penalty,
        "method": args.method,
        "nfolds": args.nfolds,
        "factor": args.factor,
        "lmbd": lmbd,
        "datafit_params": {
            k: v for k, v in f.params_to_dict().items() if k != "y"
        },
        "penalty_params": h.params_to_dict(),
    }

    print(f"  writing results...")
    file_exists = OUTPUT_PATH.exists()
    with open(OUTPUT_PATH, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
