Numerical experiments for L0-norm problems
==========================================

[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/License-AGPL--v3-red.svg)](https://github.com/TheoGuyard/El0ps/blob/main/LICENSE)

This repository contains numerical experiments related to the following papers:

- Paper 1: A New Branch-and-Bound Pruning Framework for L0-Regularized Problems. C. Elvira, T. Guyard, C. Herzet. Submitted, 2025. [arXiv](https://arxiv.org/abs/2406.03504)
- Paper 2: "A Generic Branch-and-Bound Algorithm for L0-Penalized Problems". T. Guyard, C. Elvira, C. Herzet. ICML, 2024. [arXiv](https://arxiv.org/abs/2506.03974)


## Reproducing experiments

Experiments can be easily reproduced from notebooks:

- Paper 1: [Section 4.1](notebooks/icml_2024/section_4.1.ipynb) [Sections 4.2](notebooks/icml_2024/section_4.2.ipynb)
- Paper 2: [Section 6.1](notebooks/mpc_2025/section_6.1.ipynb) [Sections 6.2 and 6.3](notebooks/mpc_2025/section_6.2_6.3.ipynb)

Alternatively, the full scripts that were used to run experiments in batches on HPC clusters are available [here](l0exp/), along with all utilities functions that are used in the notebooks.
See the [Experiment workflow](#experiment-workflow) section for more details on how to run these scripts.


## Installation

To use this repository, first clone it, enter the root folder, and install the requirements.


```bash
git clone https://github.com/TheoGuyard/l0exp
cd l0exp
pip install -e .
```

Several third-party libraries are also required to run the experiments.

### R package `L0Learn`

Some parts of the experiments require external calls to the [R](https://www.r-project.org/) package [L0Learn](https://cran.r-project.org/web/packages/L0Learn/index.html) to calibrate problem hyperparameters. It can be installed from a R console as follows:
```R
install.packages("L0Learn")
```


### Optimization solvers 

Experiments can be run using different solvers, some of which need to be installed separately by the user:
* The `l0bnb` solver should run out of the box after installing the requirements.
* The `cplex`, `gurobi` and `mosek` solvers are commercial and require a valid license to be used. Academics can obtain one for free. We refer to the [Cplex](https://www.ibm.com/products/ilog-cplex-optimization-studio), [Gurobi](https://www.gurobi.com) and [Mosek](https://www.mosek.com) websites for more information on how to obtain and set up the license.
* The `oa` solver uses `gurobi` as a backend, and has similar installation requirements.
* The `mimosa` solver needs to be installed by following the instructions on its [Gitlab page](https://gitlab.univ-nantes.fr/ls2n-sims/mimosa-solver/-/tree/samaingw?ref_type=heads). Then, users need to export a `MIMOSA_PATH` environment variable pointing to the `mimosa` executable, as well as a `MIMOSA_TMP` environment variable pointing to a temporary directory where `mimosa` can write temporary files. For instance, on Linux or MacOS, you can add the following lines to your `.bashrc` file (which needs to be sourced or restarted to take effect):
```bash
export MIMOSA_PATH="/path/to/mimosa/executable"
export MIMOSA_TMP="/path/to/temporary/directory"
```

If you encounter any problem during installation, please refer to the [Troubleshooting](#troubleshooting) section or create an [issue](https://github.com/TheoGuyard/l0exp/issues).

### Troubleshootings with `l0bnb`

- The [l0bnb](https://github.com/alisaab/l0bnb/tree/master) package cannot be installed with Python 3.11+. It is recommended to use Python 3.10 for this package.

- The [l0bnb](https://github.com/alisaab/l0bnb/tree/master) package may throw errors due to the inappropriate use of `np.Inf` instead of `np.inf` in one of its file. This can be fixed by replacing `np.Inf` with `np.inf` at file 7 of the file `l0bnb/relaxation/_utils.py` in the package source.

## Experiment workflow

The full scripts that were used to run experiments in batches on HPC clusters are available [here](l0exp/) and can be used as follows:
```bash
python -m l0exp/run.py --config <run_config_path> --save
python -m l0exp/plot.py --config <plot_config_path>
```
The parameters `<run_config_path>` and `<plot_config_path>` are paths to [yaml](https://docs.ansible.com/projects/ansible/latest/reference_appendices/YAMLSyntax.html) files specifying the experiment configuration.
The `--save` flag saves results of a run into a [pickle](https://docs.python.org/3/library/pickle.html) file under the [results](l0exp/results) folder.
The plotting script searches for all pickle files in this folder that match the configuration parameters to construct a figure.
Templates of configuration files for running and plotting experiments are available [here (run)](l0exp/configs/run_template.yml) and [here (plot)](l0exp/configs/plot_template.yml).


