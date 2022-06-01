# Imports.
import torch
import numpy as np
import ot
from tqdm import tqdm


def wasserstein_map(
    A: torch.Tensor,
    C: torch.Tensor,
    dtype: torch.dtype,
    device: str,
    R: torch.Tensor = None,
    tau: float = 0,
    progress_bar: bool = False,
) -> torch.Tensor:
    """This function maps a ground cost to the Wasserstein distance matrix on a certain dataset using that ground cost. R is an added regularization.

    Args:
        A (torch.Tensor): The input dataset, rows as samples.
        C (torch.Tensor): the ground cost.
        dtype (torch.dtype): The dtype.
        device (str): The device.
        R (torch.Tensor): The regularization matrix. Defaults to None.
        tau (float): The regularization parameter. Defaults to 0.
        progress_bar (bool): Whether to show a progress bar during the computation. Defaults to False

    Returns:
        torch.Tensor: The Wasserstein distance matrix with regularization.
    """

    # Perform some sanity checks.
    assert tau >= 0 # a positive regularization

    # Name the dimensions of the dataset (samples x features).
    n_samples, n_features = A.shape

    # Create an empty distance matrix to be populated.
    D = torch.zeros(n_samples, n_samples, dtype=dtype, device=device)

    # Initialize the progress bar, if we want one.
    if progress_bar:
        pbar = tqdm(total=n_samples * (n_samples - 1) // 2, leave=False)

    # Iterate over the lines.
    for i in range(1, n_samples):

        # Update the progress bar if it exists.
        if progress_bar:
            pbar.update(i)

        # Compute the Wasserstein distances between i,j for j < i.
        wass = ot.emd2(A[i].contiguous(), A[:i].T.contiguous(), C)

        # Add them in the distance matrix (including symmetric values).
        D[i, :i] = D[:i, i] = torch.Tensor(wass)

    # Close the progress bar if it exists.
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
    eps: float,
    dtype: torch.dtype,
    device: str,
    R: torch.Tensor = None,
    tau: float = 0,
    progress_bar: bool = False,
    stop_threshold: float = 1e-5,
    num_iter_max: int = 500,
) -> torch.Tensor:
    """This function maps a ground cost to the pairwise Sinkhorn divergence matrix on a certain dataset using that ground cost. R is an added regularization.

    Args:
        A (torch.Tensor): The input dataset, rows as samples.
        C (torch.Tensor): The ground cost.
        eps (float): The entropic regularization parameter.
        dtype (torch.dtype): The dtype.
        device (str): The device.
        R (torch.Tensor): The added regularization. Defaults to None.
        tau (float): The regularization parameter. Defaults to 0.
        progress_bar (bool): Whether to show a progress bar during the computation. Defaults to False.
        stop_threshold (float, optional): Stopping threshold for Sinkhorn (please refer to POT). Defaults to 1e-5.
        num_iter_max (int, optional): Maximum number of Sinkhorn iterations (please refer to POT). Defaults to 500.

    Returns:
        torch.Tensor: The pairwise Sinkhorn divergence matrix.
    """

    # Perform some sanity checks.
    assert tau >= 0 # a positive regularization
    assert eps >= 0 # a positive entropic regularization

    # Name the dimensions of the dataset (samples x features).
    n_samples, n_features = A.shape

    # Create an empty distance matrix to be populated.
    D = torch.zeros(n_samples, n_samples, dtype=dtype, device=device)

    # Compute the kernel.
    K = (-C / eps).exp()

    if progress_bar:
        pbar = tqdm(total=n_samples * (n_samples - 1) // 2, leave=False)

    # Iterate over the source samples.
    for i in range(n_samples):

        # Iterate over batches of target samples.
        for ii in np.array_split(range(i + 1), max(1, i // 100)):

            # Compute the Sinkhorn dual variables.
            _, wass_log = ot.sinkhorn(
                A[i].contiguous(),  # This is the source histogram.
                A[ii].T.contiguous(),  # These are the target histograms.
                C,  # This is the ground cost.
                eps,  # This is the regularization parameter.
                log=True,  # Return the dual variables
                stopThr=stop_threshold,
                numItermax=num_iter_max,
            )

            # Compute the exponential dual potentials.
            f, g = eps * wass_log["u"].log(), eps * wass_log["v"].log()

            # Compute the Sinkhorn costs.
            # These will be used to compute the Sinkhorn divergences
            wass = (
                f * A[[i] * len(ii)].T
                + g * A[ii].T
                - eps * wass_log["u"] * (K @ wass_log["v"])
            ).sum(0)

            # Add them in the distance matrix (including symmetric values).
            D[i, ii] = D[ii, i] = wass

            # Update the progress bar if it exists.
            if progress_bar:
                pbar.update(len(ii))

    # Close the progress bar if it exists.
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
    sample_prop: float,
    gamma: float,
    dtype: torch.dtype,
    device: str,
    R: torch.Tensor = None,
    tau: float = 0,
    progress_bar: bool = False,
    return_indices: bool = False,
) -> torch.Tensor:
    """Returns the stochastic Wasserstein map, updating only a random subset of
    indices and leaving the other ones as they are.

    Args:
        A (torch.Tensor): The input dataset.
        D (torch.Tensor): The intialization of the distance matrix
        C (torch.Tensor): The ground cost
        sample_prop (float): The proportion of indices to update
        gamma (float): A scaling factor
        dtype (torch.dtype): The dtype
        device (str): The device
        R (torch.Tensor): The regularization matrix. Defaults to None.
        tau (float): The regularization parameter. Defaults to 0.
        progress_bar (bool): Whether to show a progress bar during the computation. Defaults to False.
        return_indices (bool): Whether to return the updated indices. Defaults to False.
        stop_threshold (float, optional): Stopping threshold for Sinkhorn (please refer to POT). Defaults to 1e-5.
        num_iter_max (int, optional): Maximum number of Sinkhorn iterations (please refer to POT). Defaults to 500.

    Returns:
        torch.Tensor: The stochastically updated distance matrix.
    """

    # Perform some sanity checks.
    assert tau >= 0 # a positive regularization
    assert 0 < sample_prop <= 1 # a valid proportion

    # Check that input parameters make sense.
    assert gamma > 0
    assert tau >= 0

    # Name the dimensions of the dataset (samples x features).
    n_samples, n_features = A.shape

    # Define the sample size from the proportion.
    sampling_size = max(2, int(np.sqrt(sample_prop) * n_samples))

    # The indices to sample from.
    ii = np.random.choice(range(n_samples), size=sampling_size, replace=False)

    # Initialize a new distance matrix.
    D_new = D.clone()

    # Create the progress bar if we want one.
    if progress_bar:
        pbar = tqdm(total=sampling_size * (sampling_size - 1) // 2, leave=False)

    # Iterate over random indices.
    for k in range(1, sampling_size):

        # Update the progress bar if we have one.
        if progress_bar:
            pbar.update(k)

        # Compute the Wasserstein distances.
        wass = torch.Tensor(
            ot.emd2(A[ii[k]].contiguous(), A[ii[:k]].T.contiguous(), C)
        ).to(dtype=dtype, device=device)

        # Add them in the distance matrix (including symmetric values).
        # Regularization will be added later.
        D_new[ii[k], ii[:k]] = D_new[ii[:k], ii[k]] = wass

    # Close the progress bar if it exists.
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
    sample_prop: float,
    gamma: float,
    eps: float,
    R: torch.Tensor = None,
    tau: float = 0,
    progress_bar: bool = False,
    return_indices: bool = False,
    batch_size: int = 50,
    stop_threshold: float = 1e-5,
    num_iter_max: int = 100,
) -> torch.Tensor:
    """Returns the stochastic Sinkhorn divergence map, updating only a random
    subset of indices and leaving the other ones as they are.

    Args:
        A (torch.Tensor): The input dataset.
        D (torch.Tensor): The intialization of the distance matrix
        C (torch.Tensor): The ground cost
        sample_prop (float): The proportion of indices to update
        gamma (float): Rescaling parameter. In practice, one should rescale by an approximation of the singular value.
        eps (float): The entropic regularization parameter
        R (torch.Tensor): The regularization matrix. Defaults to None.
        tau (float): The regularization parameter. Defaults to 0.
        progress_bar (bool): Whether to show a progress bar during the computation. Defaults to False.
        return_indices (bool): Whether to return the updated indices. Defaults to False.
        batch_size (int): Batch size, i.e. how many distances to compute at the same time. Depends on your available GPU memory. Defaults to 50.

    Returns:
        torch.Tensor: The stochastically updated distance matrix.
    """

    # Perform some sanity checks.
    assert tau >= 0 # a positive regularization
    assert 0 < sample_prop <= 1 # a valid proportion
    assert eps >= 0 # a positive entropic regularization

    # Name the dimensions of the dataset (samples x features).
    n_samples, n_features = A.shape

    # Define the sample size from the proportion.
    sampling_size = max(2, int(np.sqrt(sample_prop) * n_samples))

    # Random indices.
    idx = np.random.choice(range(n_samples), size=sampling_size, replace=False)

    # Initialize new distance
    D_new = D.clone()

    # Compute the kernel.
    K = (-C / eps).exp()

    # Initialize the progress bar if we want one.
    if progress_bar:
        pbar = tqdm(total=sampling_size * (sampling_size - 1) // 2, leave=False)

    # Iterate over random indices.
    for k in range(sampling_size):

        i = idx[k]

        for ii in np.array_split(idx[: k + 1], max(1, k // batch_size)):

            # Compute the Sinkhorn dual variables.
            _, wass_log = ot.sinkhorn(
                A[i].contiguous(),  # This is the source histogram.
                A[ii].T.contiguous(),  # These are the target histograms.
                C,  # This is the ground cost.
                eps,  # This is the entropic regularization parameter.
                log=True,  # Return the dual variables.
                stopThr=stop_threshold,
                numItermax=num_iter_max,
            )

            # Compute the exponential dual variables.
            f, g = eps * wass_log["u"].log(), eps * wass_log["v"].log()

            # Compute the Sinkhorn costs.
            # These will be used to compute the Sinkhorn divergences below.
            wass = (
                f * A[[i] * len(ii)].T
                + g * A[ii].T
                - eps * wass_log["u"] * (K @ wass_log["v"])
            ).sum(0)

            # Add them in the distance matrix (including symmetric values).
            D_new[i, ii] = D_new[ii, i] = wass

            # Update the progress bar if we have one.
            if progress_bar:
                pbar.update(len(ii))

    # Close the progress bar if we have one.
    if progress_bar:
        pbar.close()

    # Get the indices for the grid (idx,idx).
    xx, yy = np.meshgrid(idx, idx)

    # Get the diagonal terms OT_eps(a, a)
    d = torch.diagonal(D_new[xx, yy])

    # Sinkhorn divergence OT(a, b) - (OT(a, a) + OT(b, b))/2
    D_new[xx, yy] = D_new[xx, yy] - 0.5 * (d.view(-1, 1) + d.view(1, -1))

    # Make sure there are no negative values.
    assert (D_new < 0).sum() == 0

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
