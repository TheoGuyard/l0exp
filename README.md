Numerical experiments for L0-norm problems
==========================================

[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/downloads/release/python-3100/)


## Setup

```bash
git clone https://github.com/TheoGuyard/code_2025_bnb-l0problem
cd code_2025_bnb-l0problem
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

## Known issues


### Using `l0learn` on MacOS with `arm64` architecture

The `l0learn` package can raise errors on MacOS with `arm64` architecture. To fix this, you can install a python version compiled with Rosetta for `x86_64` architecture using [pyenv](https://github.com/pyenv/pyenv). You can proceed as follows.
1. Install `pyenv`
```bash
brew install pyenv
```
2. Install a python version with Rosetta for `x86_64` architecture. The version `3.10.0` should work.
```bash
arch -x86_64 pyenv install 3.10.14
```
3. Set the python version to use
```bash
pyenv global 3.10.14
python -V
> 3.10.14
```
4. Install `l0learn` with this version of python
```bash
python -m venv .venv
. .venv/bin/activate
(.venv) pip install l0learn
```

## Contribute

Feel free to contribute by report any bug on the [issue](https://github.com/TheoGuyard/L0Exp/issues) page or by opening a [pull request](https://github.com/TheoGuyard/L0Exp/pulls).
Any feedback or contribution is welcome.
