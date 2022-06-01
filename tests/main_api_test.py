from context import wsingular
from context import utils
from context import distance

import torch

# Define the dtype and device to work with.
dtype = torch.double
device = "cpu"

# Define the dimensions of our problem.
n_samples = 15
n_features = 20

# Initialize an empty dataset.
dataset = torch.zeros((n_samples, n_features), dtype=dtype)

# Iterate over the features and samples.
for i in range(n_samples):
    for j in range(n_features):

        # Fill the dataset with translated histograms.
        dataset[i, j] = i / n_samples - j / n_features
        dataset[i, j] = torch.abs(dataset[i, j] % 1)

# Take the distance to 0 on the torus.
dataset = torch.min(dataset, 1 - dataset)

# Make it a guassian.
dataset = torch.exp(-(dataset**2) / 0.1)


def test_wasserstein_singular_vectors():

    # Compute the WSV.
    C, D = wsingular.wasserstein_singular_vectors(
        dataset,
        n_iter=10,
        dtype=dtype,
        device=device,
    )

    # Assert positivity of C.
    assert torch.sum(C < 0) == 0

    # Assert positivity of D.
    assert torch.sum(D < 0) == 0


def test_sinkhorn_singular_vectors():

    # Compute the SSV.
    C, D = wsingular.sinkhorn_singular_vectors(
        dataset,
        eps=5e-2,
        dtype=dtype,
        device=device,
        n_iter=10,
        progress_bar=True,
    )

    # Assert positivity of C.
    assert torch.sum(C < 0) == 0

    # Assert positivity of D.
    assert torch.sum(D < 0) == 0


def test_stochastic_wasserstein_singular_vectors():

    # Compute the WSV.
    C, D = wsingular.stochastic_wasserstein_singular_vectors(
        dataset,
        n_iter=20,
        dtype=dtype,
        device=device,
    )

    # Assert positivity of C.
    assert torch.sum(C < 0) == 0

    # Assert positivity of D.
    assert torch.sum(D < 0) == 0


def test_stochastic_sinkhorn_singular_vectors():

    # Compute the SSV.
    C, D = wsingular.stochastic_sinkhorn_singular_vectors(
        dataset,
        eps=5e-2,
        dtype=dtype,
        device=device,
        n_iter=20,
        progress_bar=True,
    )

    # Assert positivity of C.
    assert torch.sum(C < 0) == 0

    # Assert positivity of D.
    assert torch.sum(D < 0) == 0
