#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wsingular",
    version="0.1.6",
    description="Wasserstein Singular Vectors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Geert-Jan Huizing, Laura Cantini, Gabriel Peyré",
    url="https://github.com/gjhuizing/wsingular",
    author_email="huizing@ens.fr",
    packages=["wsingular"],
    install_requires=[
        "torch",
        "pandas",
        "sklearn",
        "seaborn",
        "matplotlib",
        "scipy",
        "tqdm",
        "numpy",
        "pot>=0.8",
        "networkx",
    ],
)
