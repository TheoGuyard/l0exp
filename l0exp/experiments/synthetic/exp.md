# `synthetic` experiment

This experiment corresponds to:
- Section 4.1 in "A New Branch-and-Bound Pruning Framework for L0-Regularized Problems" [[arXiv](https://arxiv.org/abs/2406.03504)] 


The parameters of the configuration file are as follows:
- `dataset`
  - `k`: Number of non-zero in the ground truth vector.
  - `m`: Number of rows in the observation matrix.
  - `n`: Number of columns in the observation matrix.
  - `r`: Correlation coefficient in the observation matrix.
  - `s`: Signal-to-noise ratio (in dB)
  - `seed`: Seed to use in the instance generation.
- `penalty`: Penalty function to use. See El0ps' [documentation](https://theoguyard.github.io/El0ps/html/api/penalty.html) for the list of available ones.
- `calibration`
  - `kwargs`
    - `time_limit`: Time limit in seconds for each calibration run.
- `solvers`
  - `solver-name`
    - `time_limit`: Time limit in seconds for each solver.
    - `relative_gap`: Relative gap targeted by the solver.
    - `verbose`: Toggle the verbosity of the solver.

Please do not modify the other parameters of the configuration file.


