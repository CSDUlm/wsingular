# Imports.
import torch
import numpy as np
import pandas as pd
from typing import Iterable, Tuple
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def random_distance(size: int, dtype: str, device: str) -> torch.Tensor:
    """Return a random distance-like matrix, i.e. symmetric with zero diagonal.
    The matrix is also divided by its maximum, so as to have infty norm 1.

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
    A: torch.Tensor, p: int, dtype: str, device: str
) -> torch.Tensor:
    """Return the regularization matrix [|a_i - a_j|_p]_ij

    Args:
        A (torch.Tensor): The dataset, with samples as columns
        p (int): order of the norm
        dtype (str): The dtype to be returned
        device (str): The device to be returned

    Returns:
        torch.Tensor: The regularization matrix
    """
    if p == "one":
        return 1 - torch.eye(A.shape[1]).to(dtype=dtype, device=device)
    else:
        return torch.cdist(A.T, A.T, p=p).to(dtype=dtype, device=device)


def hilbert_distance(D_1: torch.Tensor, D_2: torch.Tensor) -> float:
    """Compute the Hilbert distance between two distance-like matrices.

    Args:
        D_1 (torch.Tensor): The first matrix
        D_2 (torch.Tensor): The second matrix

    Returns:
        float: The distance
    """
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
    """Normalize the dataset and return the normalized dataset A and the
    transposed dataset B.

    Args:
        dataset (torch.Tensor): The input dataset.
        normalization_steps (int, optional): The number of Sinkhorn
        normalization steps. For large numbers, we get bistochastic matrices.
        TODO: check that. Defaults to 1 and should be larger or equal to 1.
        small_value (float): Small addition to the dataset to avoid numerical
        errors while computing OT distances. Defaults to 1e-6.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The normalized matrices A and B
    """

    # Check that `normalization_steps` > 0:
    assert normalization_steps > 0

    # Do a first normalization pass for A
    A = dataset / dataset.sum(0)
    A += small_value
    A /= A.sum(0)

    # Do a first normalization pass for B
    B = dataset.T / dataset.T.sum(0)
    B += small_value
    B /= B.sum(0)

    # Make any additional normalization steps.
    for _ in range(normalization_steps - 1):
        A, B = B.T / B.T.sum(0), A.T / A.T.sum(0)

    return A.to(dtype=dtype, device=device), B.to(dtype=dtype, device=device)


def check_uniqueness(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dtype: str,
    device: str,
) -> bool:
    # TODO: check uniqueness
    pass


def silhouette(D: torch.Tensor, labels: Iterable) -> float:
    """Return the average silhouette score, given a distance matrix and labels.

    Args:
        D (torch.Tensor): Distance matrix n*n
        labels (Iterable): n labels

    Returns:
        float: The average silhouette score
    """
    return silhouette_score(D.cpu(), labels, metric="precomputed")


def viz_TSNE(
    D: torch.Tensor,
    labels: Iterable = None,
    names: Iterable = [],
    save_path: str = None,
    p=0.1,
) -> None:
    """Visualize a distance matrix using a precomputed distance matrix.

    Args:
        D (torch.Tensor): Distance matrix
        labels (Iterable, optional): The labels, if any. Defaults to None.
    """
    tsne = TSNE(
        n_components=2, random_state=0, metric="precomputed", square_distances=True
    )
    embed = tsne.fit_transform(D.cpu())
    df = pd.DataFrame(embed, columns=["x", "y"])
    df["label"] = labels
    sns.scatterplot(data=df, x="x", y="y", hue="label")
    if len(names) > 0:
        for i in range(df.shape[0]):
            if np.random.choice([True, False], p=(p, 1 - p)):
                plt.text(x=df.x[i] + 0.3, y=df.y[i] + 0.3, s=names[i])
    if save_path:
        plt.savefig(save_path)
    plt.close()
