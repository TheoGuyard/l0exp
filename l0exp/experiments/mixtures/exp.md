# `mixture` experiment

This experiment corresponds to:
- Section 5.3 in "A Generic Branch-and-Bound Algorithm for L0-Penalized Problems" [[arXiv](https://arxiv.org/abs/2506.03974)]

The parameters of the configuration file are as follows:
- `dataset`
  - `k`: Number of non-zero in the ground truth vector.
  - `m`: Number of rows in the observation matrix.
  - `n`: Number of columns in the observation matrix.
  - `r`: Correlation coefficient in the observation matrix.
  - `s`: Signal-to-noise ratio (in dB)
  - `distrib_name` / `distrib_params`: Distribution name and parameters for the non-zero entries of the ground truth vector. See `Mixtures.sample_distribution` in the `exp.py` file for more details. 
- `solvers`
  - `solver-name`
    - `time_limit`: Time limit in seconds for each solver.
    - `relative_gap`: Relative gap targeted by the solver.
    - `verbose`: Toggle the verbosity of the solver.

Please do not modify the other parameters of the configuration file.


