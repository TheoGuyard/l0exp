# `regpath` experiment

This experiment corresponds to:
- Section 5.2 in "A Generic Branch-and-Bound Algorithm for L0-Penalized Problems" [[arXiv](https://arxiv.org/abs/2506.03974)]
- Section 4.2 in "A New Branch-and-Bound Pruning Framework for L0-Regularized Problems" [[arXiv](https://arxiv.org/abs/2406.03504)] 

The parameters of the configuration file are as follows:
- `dataset`: Path of the dataset to use. Must be a [pickle](https://docs.python.org/3/library/pickle.html) file with extension `.pkl`.
- `datafit`: Datafit function to use. See El0ps' [documentation](https://theoguyard.github.io/El0ps/html/api/datafit.html) for the list of available ones.
- `penalty`: Penalty function to use. See El0ps' [documentation](https://theoguyard.github.io/El0ps/html/api/penalty.html) for the list of available ones.
- `calibration`
  - `kwargs`
    - `time_limit`: Time limit in seconds for each calibration run.
- `path_opts`: Keyword arguments for the path fitting. See El0ps' [documentation](https://theoguyard.github.io/El0ps/html/api/path.html) for more details.
- `solvers`
  - `solver-name`
    - `time_limit`: Time limit in seconds for each solver.
    - `relative_gap`: Relative gap targeted by the solver.
    - `verbose`: Toggle the verbosity of the solver.

Please do not modify the other parameters of the configuration file.


