#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wsingular",
    version="0.1.2",
    description="Wasserstein Singular Vectors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Geert-Jan Huizing, Laura Cantini, Gabriel Peyré",
    url="https://github.com/gjhuizing/wsingular",
    author_email="huizing@ens.fr",
    packages=["wsingular"],
    install_requires=[
        "torch",
        "protobuf<=3.20.1",
        "tensorboard",
        "pandas",
        "sklearn",
        "seaborn",
        "matplotlib",
        "scipy",
        "tqdm",
        "numpy",
        "pot>=0.8",
        "scikit-network",
    ],
)
