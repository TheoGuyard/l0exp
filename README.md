Numerical experiments for L0-norm problems
==========================================

[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/License-AGPL--v3-red.svg)](https://github.com/TheoGuyard/El0ps/blob/main/LICENSE)

This repository contains numerical experiments related to L0-norm problems expressed as 

$$\textstyle\min_{\mathbf{x} \in \mathbb{R}^{n}} f(\mathbf{A}\mathbf{x}) + \lambda\|\|\mathbf{x}\|\|_0 + h(\mathbf{x})$$

and appearing in several applications.
These problems aim at minimizing a trade off between a data-fidelity function $f$ composed with a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ and the L0-norm which counts the number of non-zeros in its argument to promote sparse solutions.
The additional penalty function $h$ can be used to enforce other desirable properties on the solutions and is involved in the construction of efficient solution methods.

These experiments are linked to the following papers:
- "El0ps: An Exact L0-regularized Problems Solver" [[arXiv](https://arxiv.org/abs/2506.06373)]
- "A Generic Branch-and-Bound Algorithm for L0-Penalized Problems" [[arXiv](https://arxiv.org/abs/2506.03974)]
- "A New Branch-and-Bound Pruning Framework for L0-Regularized Problems" [[arXiv](https://arxiv.org/abs/2406.03504)] 


## Setup

To reproduce the experiments, clone this repository.


```bash
git clone https://github.com/TheoGuyard/l0exp
cd l0exp
```

Then, create a virtual environment (optional) and install the requirements.

```bash
python -m venv .venv
. .venv/bin/activate
(.venv) pip install -e .
```

See the [Known issues](#known-issues) section if you encounter any problems during the installation.
Please create an [issue](https://github.com/TheoGuyard/l0exp/issues) if needed.

## Datasets

To ensure the light-weight portability of the repository, no datasets are included.
Users can bring their own datasets by saving them in the `l0exp/datasets` directory.
A dataset must be a [pickle](https://docs.python.org/3/library/pickle.html) file with extension `.pkl` with the following keys:

- `'A'`: A feature matrix as a 2D [numpy](https://numpy.org) array of shape `(m, n)`.
- `'y'`: A target vector as a 1D [numpy](https://numpy.org) array of shape `(m,)`.

Datasets must not contain any nan, inf, or missing values.

## Available experiments

Experiments correspond to each folder in the `l0exp/experiments` directory.
An experiment contains the following files and directories:
- The `exp.md` file which contains a description of the experiment.
- The `exp.py` file which contains the main script of the experiment.
- The `exp.yaml` file which contains the configuration of the experiment to run.
- The `results` directory which contains the results of the experiment runs.
- The `saves` directory which contains the saved plots and data.

For each experiment, we refer to the `exp.md` file in the corresponding folder for more information on the purpose of the experiment and the available configurations parameters.

## Running experiments

An experiment can be run as using the command
```bash
python l0exp <expname> run
```
and its results can be plotted using the command
```bash
python l0exp <expname> plot
```
The `<expname>` keyword must match the name of one of the folders containing the experiments (e.g., `synthetic`).
The `plot` command will only recover results with a matching configuration file.

By default, these commands will consider the configuration file, results directory, and saves directory that are contained in the experiment folder.
Different ones can be specified using the following command line options:

- `-c/--config_file`: Path to the configuration file to use.
- `-r/--results_dir`: Path to the results directory to use.
- `-s/--saves_dir`: Path to the saves directory to use.

Other command line options available are as follows:

- `-n/--repeats`: Number of times to repeat the experiment. Defaults to 1.
- `-v/--verbose`: Toggle verbosity.

## Third-party solvers 

Experiments require running several solvers.
Some of them need to be installed separately by the user:
* The `mip` solver requires a valid Gurobi license that [gurobipy](https://docs.gurobi.com/current/) can access. Academics can obtain one for free. We refer to the Gurobi [website](https://www.gurobi.com/academia/academic-program-and-licenses/) for more information on how to obtain and set up the license.
* The `oa` solver uses Gurobi as a backend, and thus has similar requirements regarding licensing.
* The `mimosa` solver needs to be installed by following the instructions on its [Gitlab page](https://gitlab.univ-nantes.fr/ls2n-sims/mimosa-solver/-/tree/samaingw?ref_type=heads). Then, users need to bind the `mimosa` executable through a `MIMOSA_PATH` environment variable, as well as a temporary directory where `mimosa` can write temporary files through a `MIMOSA_TMP` environment variable.

All the other solvers should run out of the box without any additional user action.
Please create an [issue](https://github.com/TheoGuyard/l0exp/issues) if you encounter any problem.

## Known issues

- The [l0bnb](https://github.com/alisaab/l0bnb/tree/master) package cannot be install with Python 3.11+. It is recommended to use Python 3.10 for this package.

- The [l0bnb](https://github.com/alisaab/l0bnb/tree/master) package may throw errors due to the use of `np.Inf` in one of its file, instead of `np.inf`. This can be fixed by replacing `np.Inf` with `np.inf` at file 7 of the file `l0bnb/relaxation/_utils.py`.
