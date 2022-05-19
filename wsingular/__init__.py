# Imports.
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from typing import Callable, Tuple
import distance
import utils

def wasserstein_singular_vectors(
    dataset: torch.Tensor,
    tau: float,
    p: int,
    dtype: str,
    device: str,
    max_iter: int,
    writer: SummaryWriter,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.tensor = None,
    D_ref: torch.Tensor = None,
    log_loss=False,
    progress_bar=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs power iterations and return Wasserstein Singular Vectors.

    Args:
        dataset (torch.Tensor): The input dataset
        tau (float): The regularization parameter for the norm R.
        p (int): The order of the norm R
        dtype (str): The dtype.
        device (str): The device
        max_ter (int): The maximum number of power iterations.
        normalization_steps (int): How many Sinkhorn iterations for the initial
        normalization of the dataset. Must be > 1. Defaults to 1, which is just
        regular normalization, along columns for A and along rows for B. For
        large numbers of steps, A and B are bistochastic. TODO: check this.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Wasserstein sigular vectors (C,D)
    """

    # Name the dimensions of the dataset.
    m, n = dataset.shape

    # Make the transposed datasets A and B from the dataset.
    A, B = utils.normalize_dataset(
        dataset,
        normalization_steps=normalization_steps,
        small_value=small_value,
        dtype=dtype,
        device=device,
    )

    # Initialize a random cost matrix.
    C = utils.random_distance(m, dtype=dtype, device=device)
    D = utils.random_distance(n, dtype=dtype, device=device)

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)

    # Initialize loss history.
    loss_C, loss_D = [], []

    # Iterate until `max_iter`.
    for n_iter in range(max_iter):

        try:
            # Compute D using C
            D_new = distance.wasserstein_map(
                A,
                C,
                R_A,
                tau=tau,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(D_ref):
                    loss_D.append(utils.hilbert_distance(D, D_ref))
                    writer.add_scalar("Hilbert D,D_ref", loss_D[-1], n_iter)
                writer.add_scalar("Hilbert D,D_new", utils.hilbert_distance(D, D_new), n_iter)

            # Normalize D
            D = D_new / D_new.max()

            # Compute C using D
            C_new = distance.wasserstein_map(
                B,
                D,
                R_B,
                tau=tau,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(C_ref):
                    loss_C.append(utils.hilbert_distance(C, C_ref))
                    writer.add_scalar("Hilbert C,C_ref", loss_C[-1], n_iter)
                writer.add_scalar("Hilbert C,C_new", utils.hilbert_distance(C, C_new), n_iter)

            # Normalize C
            C = C_new / C_new.max()

            # TODO: Try early stopping.

        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            C /= C.max()
            D /= D.max()
            break

    if log_loss:
        return C, D, loss_C, loss_D
    else:
        return C, D


def sinkhorn_singular_vectors(
    dataset: torch.Tensor,
    tau: float,
    eps: float,
    p: int,
    dtype: str,
    device: str,
    max_iter: int,
    writer: SummaryWriter,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.tensor = None,
    D_ref: torch.Tensor = None,
    log_loss=False,
    progress_bar=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs power iterations and return Sinkhorn Singular Vectors.

    Args:
        dataset (torch.Tensor): The input dataset
        tau (float): The regularization parameter for the norm R.
        eps (float): The entropics regularization parameter.
        p (int): The order of the norm R
        dtype (str): The dtype.
        device (str): The device
        max_ter (int): The maximum number of power iterations.
        normalization_steps (int): How many Sinkhorn iterations for the initial
        normalization of the dataset. Must be > 1. Defaults to 1, which is just
        regular normalization, along columns for A and along rows for B. For
        large numbers of steps, A and B are bistochastic. TODO: check this.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sinkhorn sigular vectors (C,D)
    """

    # Name the dimensions of the dataset.
    m, n = dataset.shape

    # Make the transposed datasets A and B from the dataset U.
    A, B = utils.normalize_dataset(
        dataset,
        normalization_steps=normalization_steps,
        small_value=small_value,
        dtype=dtype,
        device=device,
    )

    # Initialize a random cost matrix.
    C = utils.random_distance(m, dtype=dtype, device=device)
    D = utils.random_distance(n, dtype=dtype, device=device)

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)

    # Initialize loss history.
    loss_C, loss_D = [], []

    # Iterate until `max_iter`.
    for n_iter in range(max_iter):

        try:

            # Compute D using C
            D_new = distance.sinkhorn_map(
                A,
                C,
                R_A,
                tau=tau,
                eps=eps,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(D_ref):
                    loss_D.append(utils.hilbert_distance(D, D_ref))
                    writer.add_scalar("Hilbert D,D_ref", loss_D[-1], n_iter)
                writer.add_scalar("Hilbert D,D_new", utils.hilbert_distance(D, D_new), n_iter)

            # Normalize D
            D = D_new / D_new.max()

            # Compute C using D
            C_new = distance.sinkhorn_map(
                B,
                D,
                R_B,
                tau=tau,
                eps=eps,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(C_ref):
                    loss_C.append(utils.hilbert_distance(C, C_ref))
                    writer.add_scalar("Hilbert C,C_ref", loss_C[-1], n_iter)
                writer.add_scalar("Hilbert C,C_new", utils.hilbert_distance(C, C_new), n_iter)

            # Normalize C
            C = C_new / C_new.max()

            # TODO: Try early stopping.

        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            C /= C.max()
            D /= D.max()
            break

    if log_loss:
        return C, D, loss_C, loss_D
    else:
        return C, D


####################### THE STOCHASTIC POWER ITERATIONS #######################


def stochastic_wasserstein_singular_vectors(
    dataset: torch.Tensor,
    tau: float,
    sample_prop: float,
    p: int,
    dtype: str,
    device: str,
    max_iter: int,
    writer: SummaryWriter,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.tensor = None,
    D_ref: torch.Tensor = None,
    progress_bar=False,
    step_fn: Callable = lambda k: 1 / np.sqrt(k),
    mult_update=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs power iterations and return Wasserstein Singular Vectors.

    Args:
        dataset (torch.Tensor): The input dataset
        tau (float): The regularization parameter for the norm R.
        sample_size (int): The number of indices to update at each step.
        p (int): The order of the norm R
        dtype (str): The dtype.
        device (str): The device
        max_ter (int): The maximum number of power iterations.
        normalization_steps (int): How many Sinkhorn iterations for the initial
        normalization of the dataset. Must be > 1. Defaults to 1, which is just
        regular normalization, along columns for A and along rows for B. For
        large numbers of steps, A and B are bistochastic. TODO: check this.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Wasserstein sigular vectors (C,D)
    """

    # Name the dimensions of the dataset.
    m, n = dataset.shape

    # Make the transposed datasets A and B from the dataset U.
    A, B = utils.normalize_dataset(
        dataset,
        normalization_steps=normalization_steps,
        small_value=small_value,
        dtype=dtype,
        device=device,
    )

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)

    D = R_A.clone()
    C = R_B.clone()

    mu, lbda = 1, 1

    # Iterate until `max_iter`.
    for k in range(1, max_iter):

        try:

            # Set the decreasing step size.
            step_size = step_fn(k)
            if writer:
                writer.add_scalar("step_size", step_size, k)

            C_new, xx, yy = distance.stochastic_wasserstein_map(
                B,
                C,
                D,
                gamma=1,
                sample_prop=sample_prop,
                R=R_B,
                tau=tau,
                dtype=dtype,
                device=device,
                return_indices=True,
            )

            lbda = (1 - step_size) * lbda + step_size * torch.sum(
                C_new[xx, yy] * C[xx, yy]
            ) / torch.sum(C[xx, yy] ** 2)

            if writer:
                writer.add_scalar("lambda", lbda, k)

            C_new[xx, yy] /= lbda

            if mult_update:
                C_new = torch.exp((1 - step_size) * C.log() + step_size * C_new.log())
            else:
                C_new = (1 - step_size) * C + step_size * C_new

            C_new.fill_diagonal_(0)

            if writer:
                if torch.is_tensor(C_ref):
                    hilbert = utils.hilbert_distance(C_new, C_ref)
                    writer.add_scalar("Hilbert C,C_ref", hilbert, k)
                writer.add_scalar("Hilbert C,C_new", utils.hilbert_distance(C, C_new), k)

            C = C_new / C_new.max()

            D_new, xx, yy = distance.stochastic_wasserstein_map(
                A,
                D,
                C,
                gamma=1,
                sample_prop=sample_prop,
                R=R_A,
                tau=tau,
                dtype=dtype,
                device=device,
                return_indices=True,
            )

            mu = (1 - step_size) * mu + step_size * torch.sum(
                D_new[xx, yy] * D[xx, yy]
            ) / torch.sum(D[xx, yy] ** 2)

            if writer:
                writer.add_scalar("mu", mu, k)

            D_new[xx, yy] /= mu

            if mult_update:
                D_new = torch.exp((1 - step_size) * D.log() + step_size * D_new.log())
            else:
                D_new = (1 - step_size) * D + step_size * D_new

            D_new.fill_diagonal_(0)

            if writer:
                if torch.is_tensor(D_ref):
                    hilbert = utils.hilbert_distance(D_new, D_ref)
                    writer.add_scalar("Hilbert D,D_ref", hilbert, k)
                writer.add_scalar("Hilbert D,D_new", utils.hilbert_distance(D, D_new), k)

            D = D_new / D_new.max()

        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            C /= C.max()
            D /= D.max()
            break
    return C, D


def stochastic_sinkhorn_singular_vectors(
    dataset: torch.Tensor,
    tau: float,
    eps: float,
    sample_prop: float,
    p: int,
    dtype: str,
    device: str,
    max_iter: int,
    writer: SummaryWriter,
    small_value: float = 1e-6,
    C_ref=None,
    D_ref=None,
    normalization_steps: int = 1,
    step_fn: Callable = lambda k: 2 / (2 + np.sqrt(k)),
    progress_bar=False,
    mult_update=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs power iterations and return Sinkhorn Singular Vectors.

    Args:
        dataset (torch.Tensor): The input dataset
        tau (float): The regularization parameter for the norm R.
        sample_size (int): The number of indices to update at each step.
        eps (float): The entropic regularization parameter.
        p (int): The order of the norm R
        dtype (str): The dtype.
        device (str): The device
        max_ter (int): The maximum number of power iterations.
        normalization_steps (int): How many Sinkhorn iterations for the initial
        normalization of the dataset. Must be > 1. Defaults to 1, which is just
        regular normalization, along columns for A and along rows for B. For
        large numbers of steps, A and B are bistochastic. TODO: check this.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sinkhorn sigular vectors (C,D)
    """

    # Name the dimensions of the dataset.
    m, n = dataset.shape

    # Make the transposed datasets A and B from the dataset U.
    A, B = utils.normalize_dataset(
        dataset,
        normalization_steps=normalization_steps,
        small_value=small_value,
        dtype=dtype,
        device=device,
    )

    # Initialize a random cost matrix.
    # C = utils.random_distance(m, dtype=dtype, device=device)
    # D = utils.random_distance(n, dtype=dtype, device=device)

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)

    D = R_A.clone()
    C = R_B.clone()

    mu, lbda = 1, 1

    # Iterate until `max_iter`.
    for k in range(1, max_iter):

        try:

            # Set the decreasing step size.
            step_size = step_fn(k)
            if writer:
                writer.add_scalar("step_size", step_size, k)

            C_new, xx, yy = distance.stochastic_sinkhorn_map(
                B,
                C,
                D,
                gamma=1,
                sample_prop=sample_prop,
                R=R_B,
                tau=tau,
                eps=eps,
                dtype=dtype,
                device=device,
                return_indices=True,
                progress_bar=progress_bar,
            )

            lbda = (1 - step_size) * lbda + step_size * torch.sum(
                C_new[xx, yy] * C[xx, yy]
            ) / torch.sum(C[xx, yy] ** 2)

            if writer:
                writer.add_scalar("lambda", lbda, k)

            C_new[xx, yy] /= lbda

            if mult_update:
                C_new = torch.exp((1 - step_size) * C.log() + step_size * C_new.log())
            else:
                C_new = (1 - step_size) * C + step_size * C_new

            C_new.fill_diagonal_(0)

            if writer:
                if torch.is_tensor(C_ref):
                    hilbert = utils.hilbert_distance(C_new, C_ref)
                    writer.add_scalar("Hilbert C,C_ref", hilbert, k)
                writer.add_scalar("Hilbert C,C_new", utils.hilbert_distance(C, C_new), k)

            C = C_new / C_new.max()

            D_new, xx, yy = distance.stochastic_sinkhorn_map(
                A,
                D,
                C,
                gamma=1,
                sample_prop=sample_prop,
                R=R_A,
                tau=tau,
                eps=eps,
                dtype=dtype,
                device=device,
                return_indices=True,
                progress_bar=progress_bar,
            )

            mu = (1 - step_size) * mu + step_size * torch.sum(
                D_new[xx, yy] * D[xx, yy]
            ) / torch.sum(D[xx, yy] ** 2)

            if writer:
                writer.add_scalar("mu", mu, k)

            D_new[xx, yy] /= mu

            if mult_update:
                D_new = torch.exp((1 - step_size) * D.log() + step_size * D_new.log())
            else:
                D_new = (1 - step_size) * D + step_size * D_new

            D_new.fill_diagonal_(0)

            if writer:
                if torch.is_tensor(D_ref):
                    hilbert = utils.hilbert_distance(D_new, D_ref)
                    writer.add_scalar("Hilbert D,D_ref", hilbert, k)
                writer.add_scalar("Hilbert D,D_new", utils.hilbert_distance(D, D_new), k)

            D = D_new / D_new.max()

        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            C /= C.max()
            D /= D.max()
            break

    return C, D
