# Imports.
import torch
import numpy as np
import ot
from tqdm import tqdm


def wasserstein_map(
    A: torch.Tensor,
    C: torch.Tensor,
    R: torch.Tensor,
    tau: float,
    dtype: str,
    device: str,
    progress_bar=False,
) -> torch.Tensor:
    """This function maps a ground cost to the Wasserstein distance matrix on
    a certain dataset using that ground cost. R is an added regularization.

    Args:
        A (torch.Tensor): The input dataset.
        C (torch.Tensor): the ground cost.
        R (torch.Tensor): The regularization matrix.
        tau (float): The regularization parameter.
        dtype (str): The dtype.
        device (str): The device.

    Returns:
        torch.Tensor: The Wasserstein distance matrix with regularization.
    """

    # Name the dimensions of the dataset (features x samples).
    m, n = A.shape

    # Create an empty distance matrix to be populated.
    D = torch.zeros(n, n, dtype=dtype, device=device)

    if progress_bar:
        pbar = tqdm(total=A.shape[1] * (A.shape[1] - 1) // 2, leave=False)

    # Iterate over the lines.
    for i in range(1, A.shape[1]):

        if progress_bar:
            pbar.update(i)

        # Compute the Wasserstein distances.
        wass = ot.emd2(A[:, i].contiguous(), A[:, :i].contiguous(), C)

        # Add them in the distance matrix (including symmetric values).
        D[i, :i] = D[:i, i] = torch.Tensor(wass)

    if progress_bar:
        pbar.close()

    # If the regularization parameter is > 0, regularize.
    if tau > 0:
        D = D + tau * R

    # Return the distance matrix.
    return D


def sinkhorn_map(
    A: torch.Tensor,
    C: torch.Tensor,
    R: torch.Tensor,
    tau: float,
    eps: float,
    dtype: str,
    device: str,
    progress_bar=False,
) -> torch.Tensor:
    """This function maps a ground cost to the Sinkhorn divergence matrix on
    a certain dataset using that ground cost. R is an added regularization.

    Args:
        A (torch.Tensor): The input dataset.
        C (torch.Tensor): The ground cost.
        R (torch.Tensor): The added regularization.
        tau (float): The regularization parameter for R.
        eps (float): The entropic regularization parameter.
        dtype (str): The dtype.
        device (str): The device.

    Returns:
        torch.Tensor: The divergence matrix.
    """

    # Name the dimensions of the dataset (features x samples).
    m, n = A.shape

    # Create an empty distance matrix to be populated.
    D = torch.zeros(n, n, dtype=dtype, device=device)

    K = (-C / eps).exp()

    if progress_bar:
        pbar = tqdm(total=A.shape[1] * (A.shape[1] - 1) // 2, leave=False)

    # Iterate over the lines.
    for i in range(A.shape[1]):

        for ii in np.array_split(range(i + 1), max(1, i // 100)):

            # Compute the Sinkhorn dual variables
            _, wass_log = ot.sinkhorn(
                A[:, i].contiguous(),  # This is the source histogram.
                A[:, ii].contiguous(),  # These are the target histograms.
                C,  # This is the ground cost.
                eps,  # This is the regularization parameter.
                log=True,  # Return the dual variables
                stopThr=1e-5,
                numItermax=500,
            )

            # Compute the exponential dual potentials.
            f, g = eps * wass_log["u"].log(), eps * wass_log["v"].log()

            # Compute the Sinkhorn costs.
            # These will be used to compute the Sinkhorn divergences
            wass = (
                f * A[:, [i] * len(ii)]
                + g * A[:, ii]
                - eps * wass_log["u"] * (K @ wass_log["v"])
            ).sum(0)

            # Add them in the distance matrix (including symmetric values).
            D[i, ii] = D[ii, i] = wass

            if progress_bar:
                pbar.update(len(ii))

    if progress_bar:
        pbar.close()

    # Get the diagonal terms OT_eps(a, a).
    d = torch.diagonal(D)

    # The Sinkhorn divergence is OT(a, b) - (OT(a, a) + OT(b, b))/2.
    D = D - 0.5 * (d.view(-1, 1) + d.view(1, -1))

    # Make sure there are no negative values.
    assert (D < 0).sum() == 0

    # Make sure the diagonal is zero.
    D.fill_diagonal_(0)

    # If the regularization parameter is > 0, regularize.
    if tau > 0:
        D = D + tau * R

    # Return the distance matrix.
    return D


############################### STOCHASTIC MAPS ###############################


def stochastic_wasserstein_map(
    A: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    R: torch.Tensor,
    sample_prop: float,
    tau: float,
    gamma: float,
    dtype: str,
    device: str,
    progress_bar=False,
    return_indices=False,
) -> torch.Tensor:
    """Returns the stochastic Wasserstein map, updating only a random subset of
    indices and leaving the other ones as they are.

    Args:
        A (torch.Tensor): The input dataset.
        D (torch.Tensor): The intialization of the distance matrix
        C (torch.Tensor): The ground cost
        R (torch.Tensor): The regularization matrix.
        sample_size (int): The number of indices to update (they are symmetric)
        tau (float): The regularization parameter for R
        dtype (str): The dtype
        device (str): The device

    Returns:
        torch.Tensor: The stochastically updated distance matrix.
    """

    assert gamma > 0
    assert tau >= 0
    # TODO assert simplex

    # Name the dimensions of the dataset (features x samples).
    m, n = A.shape

    # Define the sample size from the proportion. TODO: Not a linear function though
    sample_size = n
    sample_size = max(2, int(np.sqrt(sample_prop) * sample_size))

    # The indices to sample from
    # Random indices.
    ii = np.random.choice(range(n), size=sample_size, replace=False)

    # Initialize new distance
    D_new = D.clone()

    if progress_bar:
        pbar = tqdm(total=sample_size * (sample_size - 1) // 2, leave=False)

    # Iterate over random indices.
    for k in range(1, sample_size):

        if progress_bar:
            pbar.update(k)

        # Compute the Wasserstein distances.
        wass = torch.Tensor(
            ot.emd2(A[:, ii[k]].contiguous(), A[:, ii[:k]].contiguous(), C)
        ).to(dtype=dtype, device=device)

        # Add them in the distance matrix (including symmetric values).
        # Also add regularization.
        D_new[ii[k], ii[:k]] = D_new[ii[:k], ii[k]] = wass

    if progress_bar:
        pbar.close()

    # Make sure the diagonal is zero.
    D_new.fill_diagonal_(0)

    # Get the indices for the grid (ii,ii).
    xx, yy = np.meshgrid(ii, ii)

    # If the regularization parameter is > 0, regularize.
    if tau > 0:
        D_new[xx, yy] += tau * R[xx, yy]

    # Divide gamma
    D_new[xx, yy] /= gamma

    # Return the distance matrix.
    if return_indices:
        return D_new, xx, yy
    else:
        return D_new


def stochastic_sinkhorn_map(
    A: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    R: torch.Tensor,
    sample_prop: float,
    tau: float,
    gamma: float,
    eps: float,
    dtype: str,
    device: str,
    progress_bar=False,
    return_indices=False,
    batch_size=50,
) -> torch.Tensor:
    """Returns the stochastic Sinkhorn divergence map, updating only a random
    subset of indices and leaving the other ones as they are.

    Args:
        A (torch.Tensor): The input dataset.
        D (torch.Tensor): The intialization of the distance matrix
        C (torch.Tensor): The ground cost
        R (torch.Tensor): The regularization matrix.
        sample_size (int): The number of indices to update (they are symmetric)
        tau (float): The regularization parameter for R
        eps (float): The entropic regularization parameter
        dtype (str): The dtype
        device (str): The device

    Returns:
        torch.Tensor: The stochastically updated distance matrix.
    """

    # Name the dimensions of the dataset (features x samples).
    m, n = A.shape

    # Define the sample size from the proportion. TODO: Not a linear function though
    sample_size = n
    sample_size = max(2, int(np.sqrt(sample_prop) * sample_size))

    # Random indices.
    idx = np.random.choice(range(n), size=sample_size, replace=False)

    # Initialize new distance
    D_new = D.clone()

    K = (-C / eps).exp()

    if progress_bar:
        pbar = tqdm(total=sample_size * (sample_size - 1) // 2, leave=False)

    # Iterate over random indices.
    for k in range(sample_size):

        i = idx[k]
        # ii = idx[:k+1]

        for ii in np.array_split(idx[: k + 1], max(1, k // batch_size)):

            # Compute the Sinkhorn dual variables.
            _, wass_log = ot.sinkhorn(
                A[:, i].contiguous(),  # This is the source histogram.
                A[:, ii].contiguous(),  # These are the target histograms.
                C,  # This is the gruond cost.
                eps,  # This is the entropic regularization parameter.
                log=True,  # Return the dual variables.
                stopThr=1e-5,
                numItermax=100,
            )

            # Compute the exponential dual variables.
            f, g = eps * wass_log["u"].log(), eps * wass_log["v"].log()

            # Compute the Sinkhorn costs.
            # These will be used to compute the Sinkhorn divergences below.
            wass = (
                f * A[:, [i] * len(ii)]
                + g * A[:, ii]
                - eps * wass_log["u"] * (K @ wass_log["v"])
            ).sum(0)

            # Add them in the distance matrix (including symmetric values).
            D_new[i, ii] = D_new[ii, i] = wass

            if progress_bar:
                pbar.update(len(ii))

    if progress_bar:
        pbar.close()

    # Get the indices for the grid (idx,idx).
    xx, yy = np.meshgrid(idx, idx)

    # Get the diagonal terms OT_eps(a, a)
    d = torch.diagonal(D_new[xx, yy])

    # Sinkhorn divergence OT(a, b) - (OT(a, a) + OT(b, b))/2
    D_new[xx, yy] = D_new[xx, yy] - 0.5 * (d.view(-1, 1) + d.view(1, -1))

    # Make sure there are no negative values.
    # assert((D_new < 0).sum() == 0)

    # Make sure the diagonal is zero.
    D_new[xx, yy].fill_diagonal_(0)

    # If the regularization parameter is > 0, regularize.
    if tau > 0:
        D_new[xx, yy] += tau * R[xx, yy]

    # Divide gamma
    D_new[xx, yy] /= gamma

    # Return the distance matrix.
    if return_indices:
        return D_new, xx, yy
    else:
        return D_new