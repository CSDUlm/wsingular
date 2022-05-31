Wasserstein Singular Vectors
=======================================

.. toctree::
   :hidden:
   :maxdepth: 1
   :glob:
   :caption: Getting started

   vignettes/*

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Main API

   wsingular

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Submodules

   wsingular.distance
   wsingular.utils

|PyPI version| |Tests| |codecov| |Documentation Status|

.. |PyPI version| image:: https://img.shields.io/pypi/v/wsingular
   :target: https://pypi.org/project/wsingular/

.. |Tests| image:: https://github.com/gjhuizing/wsingular/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/gjhuizing/wsingular/actions/workflows/tests.yml

.. |codecov| image:: https://codecov.io/gh/gjhuizing/wsingular/branch/main/graph/badge.svg?token=JGIN7X8NXS
   :target: https://codecov.io/gh/gjhuizing/wsingular

.. |Documentation Status| image:: https://readthedocs.org/projects/wsingular/badge/?version=latest
   :target: https://wsingular.readthedocs.io/en/latest/?badge=latest

.. raw:: html

   <br>
   <div style='text-align:center'>
   <img src="https://user-images.githubusercontent.com/30904288/171128302-c37fdafb-f951-4c90-9ddf-61b4c6cfea9e.png" alt="fig_intro" width="500"/>
   </div>
   <br>

:code:`wsingular` is the Python package for the ICML 2022 paper "Unsupervised Ground Metric Learning Using Wasserstein Singular Vectors".

*Wasserstein Singular Vectors* simultaneously compute a Wasserstein distance between *samples* and a Wasserstein distance between *features* of a dataset.
These distance matrices emerge naturally as positive singular vectors of the function mapping ground costs to pairwise Wasserstein distances.

Get started
-----------

Install the package: :code:`pip install wsingular`

Follow the tutorials in this documentation, and if you run into issue, leave an issue on the :ref:`Github repo<https://github.com/gjhuizing/wsingular>`.

Tips
-------

- We strongly encourage :code:`torch.double` precision for numerical stability.
- You can easily run the demo notebook in Google Colab! Just use 'open from Github' and add :code:`!pip install wsingular` at the beginning.
- If you want to stop the computation of singular vectors early, just hit :code:`Ctrl-C` and the function will return the result of the latest optimization step.

Citing us
---------

The conference proceedings will be out soon. In the meantime you can cite our arXiv preprint.::

    @article{huizing2021unsupervised,
      title={Unsupervised Ground Metric Learning using Wasserstein Eigenvectors},
      author={Huizing, Geert-Jan and Cantini, Laura and Peyr{\'e}, Gabriel},
      journal={arXiv preprint arXiv:2102.06278},
      year={2021}
    }