[![Tests](https://github.com/gjhuizing/wsingular/actions/workflows/tests.yml/badge.svg)](https://github.com/gjhuizing/wsingular/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/gjhuizing/wsingular/branch/main/graph/badge.svg?token=JGIN7X8NXS)](https://codecov.io/gh/gjhuizing/wsingular)
[![Documentation Status](https://readthedocs.org/projects/wsingular/badge/?version=latest)](https://wsingular.readthedocs.io/en/latest/?badge=latest)

# Wasserstein Singular Vectors

`wsingular` is the Python package for the ICML 2022 paper "Unsupervised Ground Metric Learning Using Wasserstein Singular Vectors".

*Wasserstein Singular Vectors* simultaneously compute an OT distance between *samples* and an OT distance between *features* of a dataset.
These distance matrices emerge naturally as positive singular vectors of the function mapping ground costs to pairwise OT distances.

## Get started

Install the package: `pip install wsingular`

Follow the documentation: https://wsingular.rtfd.io

## Citing us

The conference proceedings will be out soon. In the meantime you can cite our arXiv preprint.

    @article{huizing2021unsupervised,
      title={Unsupervised Ground Metric Learning using Wasserstein Eigenvectors},
      author={Huizing, Geert-Jan and Cantini, Laura and Peyr{\'e}, Gabriel},
      journal={arXiv preprint arXiv:2102.06278},
      year={2021}
    }
