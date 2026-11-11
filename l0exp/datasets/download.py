import argparse
import pathlib
import pickle
import sys

# When run as a script, this file's directory is put first on sys.path, so
# this file would shadow the `download` package imported by `libsvmdata`.
_THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path = [
    p for p in sys.path if pathlib.Path(p or ".").resolve() != _THIS_DIR
]

from l0exp.dataset import get_dataset_realworld  # noqa: E402

OUTPUT_DIR = pathlib.Path(__file__).parent
DATASETS = {
    # regression
    "drug": "regression",
    "kits": "regression",
    "riboflavin": "regression",
    # classification
    "arcene": "classification",
    "breast-cancer": "classification",
    "colon-cancer": "classification",
    "dexter": "classification",
    "dorothea": "classification",
    "gisette": "classification",
    "leukemia": "classification",
    "madelon": "classification",
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        type=str,
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        help="Name(s) of the dataset(s) to download, or 'all'.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to normalize the columns of the design matrix before "
        "saving it.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to overwrite datasets that already exist.",
    )

    args = parser.parse_args()

    names = list(DATASETS.keys()) if "all" in args.datasets else args.datasets
    names = list(dict.fromkeys(names))

    failed = []
    for name in names:
        output_path = OUTPUT_DIR / f"{name}.pkl"
        print(f"Dataset {name}...")

        if output_path.exists() and not args.overwrite:
            print(f"  already exists at {output_path}, skipping")
            continue

        print("  downloading...")
        try:
            A, y, x = get_dataset_realworld(name, normalize=args.normalize)
        except Exception as e:
            print(f"  failed: {e}")
            failed.append(name)
            continue

        print(f"  shape: {A.shape}")
        print(f"  saving to {output_path}...")
        data = {"A": A, "y": y, "x": x, "task": DATASETS[name]}
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

    if failed:
        raise SystemExit(f"Failed to download datasets: {failed}")
