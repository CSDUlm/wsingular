[![PyPI version](https://img.shields.io/pypi/v/wsingular)](https://pypi.org/project/wsingular/)
[![Tests](https://github.com/gjhuizing/wsingular/actions/workflows/tests.yml/badge.svg)](https://github.com/gjhuizing/wsingular/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/gjhuizing/wsingular/branch/main/graph/badge.svg?token=JGIN7X8NXS)](https://codecov.io/gh/gjhuizing/wsingular)
[![Documentation Status](https://readthedocs.org/projects/wsingular/badge/?version=latest)](https://wsingular.readthedocs.io/en/latest/?badge=latest)

# Wasserstein Singular Vectors

<br>
<div style='text-align:center'>
<img src="https://user-images.githubusercontent.com/30904288/171128302-c37fdafb-f951-4c90-9ddf-61b4c6cfea9e.png" alt="fig_intro" width="500"/>
</div>
<br>

`wsingular` is the Python package for the ICML 2022 paper "Unsupervised Ground Metric Learning Using Wasserstein Singular Vectors".

*Wasserstein Singular Vectors* simultaneously compute a Wasserstein distance between *samples* and a Wasserstein distance between *features* of a dataset.
These distance matrices emerge naturally as positive singular vectors of the function mapping ground costs to pairwise Wasserstein distances.

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
