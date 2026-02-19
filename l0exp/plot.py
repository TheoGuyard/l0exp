import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import pickle
import yaml

from el0ps.solver import Status


def is_subset_dict(small: dict, big: dict) -> bool:
    for k, v in small.items():
        if k not in big:
            return False
        vv = big[k]
        if isinstance(v, dict):
            if not isinstance(vv, dict):
                return False
            if not is_subset_dict(v, vv):
                return False
        else:
            if v != "ANY" and v != vv:
                return False
    return True


def get_graphics_data(config: dict):

    raw_results = []
    results_dir = pathlib.Path(__file__).parent / "results"
    results_fmt = "{}_*.pkl".format(config["experiment"])
    for result_path in pathlib.Path(results_dir).glob(results_fmt):
        with open(result_path, 'rb') as result_file:
            result_data = pickle.load(result_file)

            result_match = True
            
            if result_data["config"]["experiment"] != config["experiment"]:
                result_match = False

            if not is_subset_dict(result_data["config"]["dataset"], config["dataset"]):
                result_match = False

            if not is_subset_dict(result_data["config"]["calibration"], config["calibration"]):
                result_match = False

            if not is_subset_dict(result_data["config"]["action"], config["action"]):
                result_match = False
            
            for _, solver_config in result_data["config"]["solvers"].items():
                if config["solvers"]["type"] != "ANY":
                    if solver_config["type"] != config["solvers"]["type"]:
                        result_match = False
                if solver_config["args"] != config["solvers"]["args"]:
                    result_match = False

            if result_match:
                raw_results.append(result_data["results"])
    
    results = {}
    for graphic, graphic_config in config["graphics"].items():
        results[graphic] = {}
        if graphic_config["type"].startswith("path_"):
            for raw_result in raw_results:
                for solver_name, solver_path in raw_result.items():
                    if solver_path is None:
                        continue
                    if solver_name not in results[graphic]:
                        results[graphic][solver_name] = {lmbd: [] for lmbd in solver_path.keys()}
                    for lmbd, path_result in solver_path.items():
                        if graphic_config["type"] == "path_time":
                            results[graphic][solver_name][lmbd].append(path_result.solve_time)
                        elif graphic_config["type"] == "path_value":
                            results[graphic][solver_name][lmbd].append(path_result.objective_value)
                        elif graphic_config["type"] == "path_nnz":
                            results[graphic][solver_name][lmbd].append(np.count_nonzero(path_result.x))
                        else:
                            raise ValueError(f"Unknown graphic type '{graphic_config['type']}'")
        elif graphic_config["type"].startswith("solve_"):
            if graphic_config["type"] == "solve_profile":
                time_grid = np.logspace(
                    np.log10(graphic_config["args"].get("time_min", 1e-4)),
                    np.log10(graphic_config["args"].get("time_max", 1e+3)),
                    graphic_config["args"].get("time_num", 70)
                )
                for raw_result in raw_results:
                    for solver_name, solver_result in raw_result.items():
                        if solver_result is None:
                            continue
                        if solver_name not in results[graphic]:
                            results[graphic][solver_name] = {time_step: [0] for time_step in time_grid}
                        if solver_result.status == Status.OPTIMAL:
                            for time_step in time_grid:
                                if solver_result.solve_time <= time_step:
                                    results[graphic][solver_name][time_step][0] += 1
                if graphic_config["args"].get("normalize", False):
                    for solver_name, solver_data in results[graphic].items():
                        tot_solved = solver_data[time_grid[-1]][0]
                        if tot_solved > 0:
                            for time_step, num_solved in solver_data.items():
                                num_solved[0] /= tot_solved
            elif graphic_config["type"] == "solve_statistics":
                for raw_result in raw_results:
                    for solver_name, solver_result in raw_result.items():
                        if solver_result is None:
                            continue
                        if solver_name not in results[graphic]:
                            results[graphic][solver_name] = {"solve_time": []}
                        if solver_result.status == Status.OPTIMAL:
                            results[graphic][solver_name]["solve_time"].append(solver_result.solve_time)
        else:
            raise ValueError(f"Unknown graphic type '{graphic_config['type']}'")
            
    return results



parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()


# Load configuration
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
print("=" * 20 + f" Plotting experiment {config['experiment']} " + "=" * 20)


# Collect results
print("Collecting results...")
graphics_data = get_graphics_data(config)

# graphics_data[graphic][label][xaxis][yaxis]

if args.save:

    for (graphic, graphic_data) in graphics_data.items():

        table = pd.DataFrame()

        # Add common xaxis
        if config["graphics"][graphic]["type"] == "path_time":
            table["lmbd_ratio"] = np.logspace(
                np.log10(config["action"]["args"]["lmbd_max"]),
                np.log10(config["action"]["args"]["lmbd_min"]),
                config["action"]["args"]["lmbd_num"]
            )
            table["limit_time"] = np.full(
                len(table),
                config["solvers"]["args"]["time_limit"]
            )
        elif config["graphics"][graphic]["type"] == "solve_profile":
            table["time"] = np.logspace(
                np.log10(config["graphics"][graphic]["args"]["time_min"]),
                np.log10(config["graphics"][graphic]["args"]["time_max"]),
                config["graphics"][graphic]["args"]["time_num"]
            )

        maxrow = max([len(values) for values in graphic_data.values()])
        maxrow = max(maxrow, len(table))
        xlabel = config["graphics"][graphic]["args"]["xlabel"]
        ylabel = config["graphics"][graphic]["args"]["ylabel"]

        for label, label_data in graphic_data.items():
            col_xlabel = label + "_" + xlabel
            col_ylabel = label + "_" + ylabel
            col_xvalue = []
            col_yvalue = []
            for xvalue, yvalues in label_data.items():
                col_xvalue.append(xvalue)
                col_yvalue.append(np.mean(yvalues))
            while len(col_xvalue) < maxrow:
                col_xvalue.append(np.nan)
                col_yvalue.append(np.nan)

            table[col_xlabel] = col_xvalue
            table[col_ylabel] = col_yvalue
        
        output_dir = pathlib.Path(__file__).parent / "saves"
        output_file = f"{config['experiment']}_{graphic}.csv"
        output_path = output_dir / output_file
        if output_path.exists():
            raise FileExistsError(f"Output file '{output_file}' already exists.")
        else:
            table.to_csv(output_path, index=False)
            print(f"Saved {graphic} graphic data to '{output_file}'")

else:


    labels = set([label for graphic in graphics_data.values() for label in graphic.keys()])
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    color_map = {label: colors[i] for i, label in enumerate(labels)}
    global_lines = []
    global_labels = []

    fig, axs = plt.subplots(1, len(graphics_data), figsize=(4 * len(graphics_data), 4))
    for i, (graphic, graphic_data) in enumerate(graphics_data.items()):
        
        ax = axs[i] if len(graphics_data) > 1 else axs

        for label, label_data in graphic_data.items():
                
            (line, ) = ax.plot(
                label_data.keys(),
                [np.mean(yvals) for _, yvals in label_data.items()],
                marker=".",
                color=color_map[label],
                label=label,
            )

            print(f",{[float(np.mean(yvals)) for _, yvals in label_data.items()][0]}", end="")

            if i == 0:
                global_lines.append(line)
                global_labels.append(label)
        
        for arg, val in config["graphics"][graphic]["args"].items():
            if arg == "xlabel":
                ax.set_xlabel(val)
            elif arg == "ylabel":
                ax.set_ylabel(val)
            elif arg == "xscale":
                ax.set_xscale(val)
            elif arg == "yscale":
                ax.set_yscale(val)
            elif arg == "invert_xaxis":
                if val:
                    ax.invert_xaxis()

        ax.grid(True)

    # fig.legend(
    #     global_lines,
    #     global_labels,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 0.9),
    #     ncol=3,
    # )
    # plt.suptitle(f"Experiment {config['experiment']}")
    # plt.tight_layout(rect=[0, 0, 1, 0.75])
    # plt.show()
