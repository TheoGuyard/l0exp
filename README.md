Numerical experiments for L0-norm problems
==========================================

[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/downloads/release/python-3100/)


## Setup

```bash
git clone https://github.com/TheoGuyard/l0exp
cd l0exp
```

```bash
python -m venv .venv
. .venv/bin/activate
(.venv) pip install -e .
```

## Running experiments

```bash
python <exp_file.py> run -c <config_file.yml> -r <results_dir> -v
```

```bash
python <exp_file.py> plot -c <config_file.yml> -r <results_dir> -s <saves_dir> -v
```
