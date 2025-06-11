# `calibration` experiment

This experiment is dedicated to the calibration of the parameters of L0-norm penalized problems with real-world datasets.
The calibration is performed by cross-validation over a grid of hyperparameters.
Each combination is solved optimally using [El0ps](https://github.com/TheoGuyard/El0ps), and the best ones are selected based on a BIC criterion.

The parameters of the configuration file are as follows:
- `dataset`: Path of the dataset to use. Must be a [pickle](https://docs.python.org/3/library/pickle.html) file with extension `.pkl`.
- `datafit`: Datafit function to use. See El0ps' [documentation](https://theoguyard.github.io/El0ps/html/api/datafit.html) for the list of available ones.
- `penalty`: Penalty function to use. See El0ps' [documentation](https://theoguyard.github.io/El0ps/html/api/penalty.html) for the list of available ones.
- `calibration`
  - `kwargs`
    - `time_limit`: Time limit in seconds for each calibration run.

Please do not modify the other parameters of the configuration file.


