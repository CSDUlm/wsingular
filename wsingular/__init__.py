# Imports.
import torch
import numpy as np
from typing import Callable, Tuple
from wsingular import distance
from wsingular import utils


def wasserstein_singular_vectors(
    dataset: torch.Tensor,
    dtype: torch.dtype,
    device: str,
    n_iter: int,
    tau: float = 0,
    p: int = 1,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    log_loss: bool = False,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs power iterations and returns Wasserstein Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset, rows as samples. Alternatively, you can give a tuple of tensors (A, B).
        dtype (str): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        p (int, optional): The order of the norm R. Defaults to 1.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        log_loss (bool, optional): Whether to return the loss. Defaults to False.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Wasserstein sigular vectors (C,D). If `log_loss`, it returns (C, D, loss_C, loss_D)
    """

    # Perform some sanity checks.
    assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
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

    # Initialize loss history.
    loss_C, loss_D = [], []

    # Iterate until `n_iter`.
    for k in range(n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:
            # Compute D using C
            D_new = distance.wasserstein_map(
                A,
                C,
                R=R_A,
                tau=tau,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(D_ref):
                    loss_D.append(utils.hilbert_distance(D, D_ref))
                    writer.add_scalar("Hilbert D,D_ref", loss_D[-1], k)
                writer.add_scalar(
                    "Hilbert D,D_new",
                    utils.hilbert_distance(D, D_new),
                    k,
                )

            # Normalize D
            D = D_new / D_new.max()

            # Compute C using D
            C_new = distance.wasserstein_map(
                B,
                D,
                R=R_B,
                tau=tau,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(C_ref):
                    loss_C.append(utils.hilbert_distance(C, C_ref))
                    writer.add_scalar("Hilbert C,C_ref", loss_C[-1], k)
                writer.add_scalar(
                    "Hilbert C,C_new",
                    utils.hilbert_distance(C, C_new),
                    k,
                )

            # Normalize C
            C = C_new / C_new.max()

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
    dtype: str,
    device: str,
    n_iter: int,
    tau: float = 0,
    eps: float = 5e-2,
    p: int = 1,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    log_loss: bool = False,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs power iterations and returns Sinkhorn Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset. Alternatively, you can give a tuple of tensors (A, B).
        dtype (str): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        eps (float): The entropic regularization parameter.
        p (int, optional): The order of the norm R. Defaults to 1.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        log_loss (bool, optional): Whether to return the loss. Defaults to False.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sinkhorn Singular Vectors (C,D). If `log_loss`, it returns (C, D, loss_C, loss_D)
    """

    # Perform some sanity checks.
    assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert eps >= 0  # a positive entropic regularization
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
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

    # Initialize loss history.
    loss_C, loss_D = [], []

    # Iterate until `n_iter`.
    for k in range(n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:

            # Compute D using C
            D_new = distance.sinkhorn_map(
                A,
                C,
                R=R_A,
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
                    writer.add_scalar("Hilbert D,D_ref", loss_D[-1], k)
                writer.add_scalar(
                    "Hilbert D,D_new", utils.hilbert_distance(D, D_new), k
                )

            # Normalize D
            D = D_new / D_new.max()

            # Compute C using D
            C_new = distance.sinkhorn_map(
                B,
                D,
                R=R_B,
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
                    writer.add_scalar("Hilbert C,C_ref", loss_C[-1], k)
                writer.add_scalar(
                    "Hilbert C,C_new", utils.hilbert_distance(C, C_new), k
                )

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


def stochastic_wasserstein_singular_vectors(
    dataset: torch.Tensor,
    dtype: torch.dtype,
    device: str,
    n_iter: int,
    tau: float = 0,
    sample_prop: float = 1e-1,
    p: int = 1,
    step_fn: Callable = lambda k: 1 / np.sqrt(k),
    mult_update: bool = False,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs stochastic power iterations and returns Wasserstein Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset.  Alternatively, you can give a tuple of tensors (A, B).
        dtype (torch.dtype): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        sample_prop (float, optional): The proportion of indices to update at each step. Defaults to 1e-1.
        p (int, optional): The order of the norm R. Defaults to 1.
        step_fn (Callable, optional): The function that defines step size from the iteration number (which starts at 1). Defaults to lambdak:1/np.sqrt(k).
        mult_update (bool, optional): If True, use multiplicative update instead of additive update. Defaults to False.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Wasserstein Singular Vectors (C,D)
    """

    # Perform some sanity checks.
    assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert 0 < sample_prop <= 1  # a valid proportion
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
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

    # Initialize the singular vectors as the regularization matrices.
    D = R_A.clone()
    C = R_B.clone()

    # Initialize the approximations of the singular values. Rescaling at each iteration
    # by the (approximated) singular value make convergence super duper fast.
    mu, lbda = 1, 1

    # Iterate until `n_iter`.
    for k in range(1, n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:

            # Set the decreasing step size.
            step_size = step_fn(k)

            # Log the step size if there is a writer.
            if writer:
                writer.add_scalar("step_size", step_size, k)

            # Update a random subset of the indices.
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
                progress_bar=progress_bar,
            )

            # Update the approximation of the singular value lambda.
            lbda = (1 - step_size) * lbda + step_size * torch.sum(
                C_new[xx, yy] * C[xx, yy]
            ) / torch.sum(C[xx, yy] ** 2)

            # Log the approximation of lambda if there is a writer.
            if writer:
                writer.add_scalar("lambda", lbda, k)

            # Rescale the updated indices by the approximation of the singular value.
            C_new[xx, yy] /= lbda

            # Update the singular vector, either multiplicatively or additively.
            if mult_update:
                C_new = torch.exp((1 - step_size) * C.log() + step_size * C_new.log())
            else:
                C_new = (1 - step_size) * C + step_size * C_new

            # Make sure the diagonal of C is 0.
            C_new.fill_diagonal_(0)

            # If we have a writer, compute some losses.
            if writer:

                # If we have a reference, compute Hilbert distance to ref.
                if torch.is_tensor(C_ref):
                    hilbert = utils.hilbert_distance(C_new, C_ref)
                    writer.add_scalar("Hilbert C,C_ref", hilbert, k)

                # Compute the Hilbert distance to the value of C at the previous step.
                writer.add_scalar(
                    "Hilbert C,C_new", utils.hilbert_distance(C, C_new), k
                )

            # Rescale the singular vector.
            C = C_new / C_new.max()

            # Update a random subset of the indices.
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
                progress_bar=progress_bar,
            )

            # Update the approximation of the singular value mu.
            mu = (1 - step_size) * mu + step_size * torch.sum(
                D_new[xx, yy] * D[xx, yy]
            ) / torch.sum(D[xx, yy] ** 2)

            # Log the approximation of mu if there is a writer.
            if writer:
                writer.add_scalar("mu", mu, k)

            # Rescale the updated indices by the approximation of the singular value mu.
            D_new[xx, yy] /= mu

            # Update the singular vector, either multiplicatively or additively.
            if mult_update:
                D_new = torch.exp((1 - step_size) * D.log() + step_size * D_new.log())
            else:
                D_new = (1 - step_size) * D + step_size * D_new

            # Make sure the diagonal of D is 0.
            D_new.fill_diagonal_(0)

            # If we have a writer, compute some losses.
            if writer:

                # If we have a reference, compute Hilbert distance to ref.
                if torch.is_tensor(D_ref):
                    hilbert = utils.hilbert_distance(D_new, D_ref)
                    writer.add_scalar("Hilbert D,D_ref", hilbert, k)

                # Compute the Hilbert distance to the value of C at the previous step.
                writer.add_scalar(
                    "Hilbert D,D_new",
                    utils.hilbert_distance(D, D_new),
                    k,
                )

            # Rescale the singular vector.
            D = D_new / D_new.max()

        # In case of Ctrl-C, make sure C and D are rescaled properly
        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            C /= C.max()
            D /= D.max()
            break

    # Return the singular vectors.
    return C, D


def stochastic_sinkhorn_singular_vectors(
    dataset: torch.Tensor,
    dtype: torch.dtype,
    device: str,
    n_iter: int,
    tau: float = 0,
    eps: float = 5e-2,
    sample_prop: float = 1e-1,
    p: int = 1,
    step_fn: Callable = lambda k: 1 / np.sqrt(k),
    mult_update: bool = False,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs stochastic power iterations and returns Sinkhorn Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset.  Alternatively, you can give a tuple of tensors (A, B).
        dtype (torch.dtype): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        eps (float, optional): The entropic regularization parameter. Defaults to 5e-2.
        sample_prop (float, optional): The proportion of indices to update at each step. Defaults to 1e-1.
        p (int, optional): The order of the norm R. Defaults to 1.
        step_fn (Callable, optional): The function that defines step size from the iteration number (which starts at 1). Defaults to lambdak:1/np.sqrt(k).
        mult_update (bool, optional): If True, use multiplicative update instead of additive update. Defaults to False.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sinkhorn Singular Vectors (C,D)
    """

    # Perform some sanity checks.
    assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert 0 < sample_prop <= 1  # a valid proportion
    assert eps >= 0  # a positive entropic regularization
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
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

    # Initialize the singular vectors as the regularization matrices.
    # Initialize the approximations of the singular values. Rescaling at each iteration
    # by the (approximated) singular value make convergence super duper fast.
    D = R_A.clone()
    C = R_B.clone()

    # Initialize the approximations of the singular values. Rescaling at each iteration
    # by the (approximated) singular value make convergence super duper fast.
    mu, lbda = 1, 1

    # Iterate until `n_iter`.
    for k in range(1, n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:

            # Set the decreasing step size.
            step_size = step_fn(k)

            # Log the step size if there is a writer.
            if writer:
                writer.add_scalar("step_size", step_size, k)

            # Update a random subset of the indices.
            C_new, xx, yy = distance.stochastic_sinkhorn_map(
                B,
                C,
                D,
                R=R_B,
                gamma=1,
                sample_prop=sample_prop,
                tau=tau,
                eps=eps,
                return_indices=True,
                progress_bar=progress_bar,
            )

            # Update the approximation of the singular value lambda.
            lbda = (1 - step_size) * lbda + step_size * torch.sum(
                C_new[xx, yy] * C[xx, yy]
            ) / torch.sum(C[xx, yy] ** 2)

            # Log the approximation of lambda if there is a writer.
            if writer:
                writer.add_scalar("lambda", lbda, k)

            # Rescale the updated indices by the approximation of the singular value.
            C_new[xx, yy] /= lbda

            # Update the singular vector, either multiplicatively or additively.
            if mult_update:
                C_new = torch.exp((1 - step_size) * C.log() + step_size * C_new.log())
            else:
                C_new = (1 - step_size) * C + step_size * C_new

            # Make sure the diagonal of C is 0.
            C_new.fill_diagonal_(0)

            # If we have a writer, compute some losses.
            if writer:
                # If we have a reference, compute Hilbert distance to ref.
                if torch.is_tensor(C_ref):
                    hilbert = utils.hilbert_distance(C_new, C_ref)
                    writer.add_scalar("Hilbert C,C_ref", hilbert, k)

                # Compute the Hilbert distance to the value of C at the previous step.
                writer.add_scalar(
                    "Hilbert C,C_new",
                    utils.hilbert_distance(C, C_new),
                    k,
                )

            # Rescale the singular vector.
            C = C_new / C_new.max()

            # Update a random subset of the indices.
            D_new, xx, yy = distance.stochastic_sinkhorn_map(
                A,
                D,
                C,
                R=R_A,
                gamma=1,
                sample_prop=sample_prop,
                tau=tau,
                eps=eps,
                return_indices=True,
                progress_bar=progress_bar,
            )

            # Update the approximation of the singular value mu.
            mu = (1 - step_size) * mu + step_size * torch.sum(
                D_new[xx, yy] * D[xx, yy]
            ) / torch.sum(D[xx, yy] ** 2)

            # Log the approximation of mu if there is a writer.
            if writer:
                writer.add_scalar("mu", mu, k)

            # Rescale the updated indices by the approximation of the singular value mu.
            D_new[xx, yy] /= mu

            # Update the singular vector, either multiplicatively or additively.
            if mult_update:
                D_new = torch.exp((1 - step_size) * D.log() + step_size * D_new.log())
            else:
                D_new = (1 - step_size) * D + step_size * D_new

            # Make sure the diagonal of D is 0.
            D_new.fill_diagonal_(0)

            # If we have a writer, compute some losses.
            if writer:

                # If we have a reference, compute Hilbert distance to ref.
                if torch.is_tensor(D_ref):
                    hilbert = utils.hilbert_distance(D_new, D_ref)
                    writer.add_scalar("Hilbert D,D_ref", hilbert, k)

                # Compute the Hilbert distance to the value of C at the previous step.
                writer.add_scalar(
                    "Hilbert D,D_new",
                    utils.hilbert_distance(D, D_new),
                    k,
                )
            # Rescale the singular vector.
            D = D_new / D_new.max()

        # In case of Ctrl-C, make sure C and D are rescaled properly
        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            C /= C.max()
            D /= D.max()
            break

    return C, D
