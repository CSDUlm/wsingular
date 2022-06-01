from context import wsingular
from context import utils
from context import distance

import torch
import numpy as np

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

def test_random_distance():

    # Generate a random distance matrix.
    D = wsingular.utils.random_distance(n_samples, dtype=dtype, device=device)

    # Assert correct dimensions of D.
    assert D.shape == (n_samples, n_samples)

    # Assert positivity of D.
    assert torch.sum(D < 0) == 0

def test_hilbert_distance():

    # Generate random distance matrices.
    D_1 = wsingular.utils.random_distance(n_samples, dtype=dtype, device=device)
    D_2 = wsingular.utils.random_distance(n_samples, dtype=dtype, device=device)

    # Compute Hilbert distance.
    dist = wsingular.utils.hilbert_distance(D_1, D_2)

    # Assert distance is strictly positive.
    assert dist > 0

def test_silhouette():

    # Generate a random distance matrix.
    D = wsingular.utils.random_distance(n_samples, dtype=dtype, device=device)

    # Dummy labels
    labels = np.random.choice([0, 1], size=n_samples)

    # Compute silhouette score.
    wsingular.utils.silhouette(D, labels)

def test_regularization_matrix():

    A, B = wsingular.utils.normalize_dataset(dataset, dtype=dtype, device=device)

    R = wsingular.utils.regularization_matrix(A, p=1, dtype=dtype, device=device)

    # Assert correct dimensions of A.
    assert A.shape == (n_samples, n_features)

    # Assert correct dimensions of R.
    assert R.shape == (n_samples, n_samples)

    # Assert positivity of A.
    assert torch.sum(A < 0) == 0

    # Assert positivity of R.
    assert torch.sum(R < 0) == 0
