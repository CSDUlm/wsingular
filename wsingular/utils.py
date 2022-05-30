# Imports.
import torch
import pandas as pd
from typing import Iterable, Tuple
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ot
from sknetwork.topology import get_connected_components


def random_distance(size: int, dtype: str, device: str) -> torch.Tensor:
    """Return a random distance-like matrix, i.e. symmetric with zero diagonal. The matrix is also divided by its maximum, so as to have infty norm 1.

    Args:
        size (int): Will return a matrix of dimensions size*size
        dtype (str): The dtype to be returned
        device (str): The device to be returned

    Returns:
        torch.Tensor: The random distance-like matrix
    """
    # Create a random matrix.
    D = torch.rand(size, size, dtype=dtype, device=device)

    # Make it symmetric.
    D = D + D.T

    # Make it zero diagonal.
    D.fill_diagonal_(0)

    # Return the normalized matrix.
    return D / D.max()


def regularization_matrix(
    A: torch.Tensor,
    p: int,
    dtype: str,
    device: str,
) -> torch.Tensor:
    """Return the regularization matrix

    Args:
        A (torch.Tensor): The dataset, with samples as rows
        p (int): order of the norm
        dtype (str): The dtype to be returned
        device (str): The device to be returned

    Returns:
        torch.Tensor: The regularization matrix
    """

    # Return the pairwise distances using torch's `cdist`.
    return torch.cdist(A, A, p=p).to(dtype=dtype, device=device)


def hilbert_distance(D_1: torch.Tensor, D_2: torch.Tensor) -> float:
    """Compute the Hilbert distance between two distance-like matrices.

    Args:
        D_1 (torch.Tensor): The first matrix
        D_2 (torch.Tensor): The second matrix

    Returns:
        float: The distance
    """

    # Perform some sanity checks.
    assert torch.sum(D_1 < 0) == 0  # positivity
    assert torch.sum(D_2 < 0) == 0  # positivity
    assert D_1.shape == D_2.shape  # same shape

    # Get a mask of all indices except the diagonal.
    idx = torch.eye(D_1.shape[0]) != 1

    # Compute the log of D1/D2 (except on the diagonal)
    div = torch.log(D_1[idx] / D_2[idx])

    # Return the Hilbert projective metric.
    return float((div.max() - div.min()).cpu())


def normalize_dataset(
    dataset: torch.Tensor,
    dtype: str,
    device: str,
    normalization_steps: int = 1,
    small_value: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize the dataset and return the normalized dataset A and the transposed dataset B.

    Args:
        dataset (torch.Tensor): The input dataset, samples as rows.
        normalization_steps (int, optional): The number of Sinkhorn normalization steps. For large numbers, we get bistochastic matrices. Defaults to 1 and should be larger or equal to 1.
        small_value (float): Small addition to the dataset to avoid numerical errors while computing OT distances. Defaults to 1e-6.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The normalized matrices A and B.
    """

    # Perform some sanity checks.
    assert len(dataset.shape) == 2  # correct shape
    assert torch.sum(dataset < 0) == 0  # positivity
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    # Do a first normalization pass for A
    A = dataset / dataset.sum(1).reshape(-1, 1)
    A += small_value
    A = A.sum(1).reshape(-1, 1)

    # Do a first normalization pass for B
    B = dataset.T / dataset.T.sum(1).reshape(-1, 1)
    B += small_value
    B /= B.sum(1).reshape(-1, 1)

    # Make any additional normalization steps.
    for _ in range(normalization_steps - 1):
        A, B = B.T / B.T.sum(1).reshape(-1, 1), A.T / A.T.sum(1).reshape(-1, 1)

    return A.to(dtype=dtype, device=device), B.to(dtype=dtype, device=device)


def check_uniqueness(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> bool:
    """Check uniqueness of singular vectors using the graph connectivity criterion described in the paper. Caution: transport plans such that the graph is connected might exist even if this returns False. If you want to check uniqueness, consider mointoring the Hilbert distance between several outcomes of the power iterations.

    Args:
        A (torch.Tensor): The samples.
        B (torch.Tensor): The features.
        C (torch.Tensor): The ground cost.
        D (torch.Tensor): The pairwise distance.

    Returns:
        bool: Whether the criterion is verified. Caution, this might be a False negative because we do not check all possible transport plans.
    """

    # Get the shapes of pairwise distance matrices.
    m, n = C.shape[0], D.shape[0]

    # Initialize the directed adjacency matrix.
    adj = np.zeros((m * m + n * n, m * m + n * n))

    # Iterate over samples.
    for i in range(n):
        for j in range(i + 1):

            # Compute the transport plan between these samples.
            P = ot.emd(A[i].contiguous(), A[j].contiguous(), C)

            # Iterate over features.
            for k in range(m):
                for l in range(m):

                    # Fill the adjacency matrix.
                    adj[k * m + l, m * m + i * n + j] = P[k, l]
                    adj[k * m + l, m * m + j * n + i] = P[k, l]

    # Iterate over features.
    for k in range(m):
        for l in range(k + 1):

            # Compute the transport plan between these features.
            P = ot.emd(B[k].contiguous(), B[l].contiguous(), D)

            # Iterate over samples.
            for i in range(n):
                for j in range(n):

                    # Fill the adjacency matrix.
                    adj[m * m + i * n + j, k * m + l] = P[i, j]
                    adj[m * m + i * n + j, l * m + m] = P[i, j]

    # Check if the graph is connected.
    weak_labels = get_connected_components(adj, connection="weak")

    return len(np.unique(weak_labels)) == 1


def silhouette(D: torch.Tensor, labels: Iterable) -> float:
    """Return the average silhouette score, given a distance matrix and labels.

    Args:
        D (torch.Tensor): Distance matrix n*n
        labels (Iterable): n labels

    Returns:
        float: The average silhouette score
    """

    # Perform some sanity checks.
    assert len(D.shape) == 2  # correct shape
    assert torch.sum(D < 0) == 0  # positivity

    return silhouette_score(D.cpu(), labels, metric="precomputed")


def viz_TSNE(D: torch.Tensor, labels: Iterable = None) -> None:
    """Visualize a distance matrix using a precomputed distance matrix.

    Args:
        D (torch.Tensor): Distance matrix
        labels (Iterable, optional): The labels, if any. Defaults to None.
    """

    # Perform some sanity checks.
    assert len(D.shape) == 2  # correct shape
    assert torch.sum(D < 0) == 0  # positivity
    assert D.shape[1] == len(labels)  # maching labels

    # Define the t-SNE model.
    tsne = TSNE(
        n_components=2,
        random_state=0,
        metric="precomputed",
        square_distances=True,
    )

    # Compute the t-SNE embedding based on precomputed pairwise distances.
    embed = tsne.fit_transform(D.cpu())

    # Turn the embedding into a DataFrame to make Seaborn happy.
    df = pd.DataFrame(embed, columns=["x", "y"])
    df["label"] = labels

    # Make a scatterplot using Seaborn, and plot it.
    sns.scatterplot(data=df, x="x", y="y", hue="label")
    plt.show()
