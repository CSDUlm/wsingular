################################### IMPORTS ###################################

# Matrices
import torch
import numpy as np

# Optimal Transport
import ot

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Typing
from typing import Tuple

############################## HELPER FUNCTIONS ###############################

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
    A: torch.Tensor, p: int, dtype: str, device: str) -> torch.Tensor:
    """Return the regularization matrix [|a_i - a_j|_p]_ij

    Args:
        A (torch.Tensor): The dataset, with samples as columns
        p (int): order of the norm
        dtype (str): The dtype to be returned
        device (str): The device to be returned

    Returns:
        torch.Tensor: The regularization matrix
    """
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
    idx = (torch.eye(D_1.shape[0]) != 1)

    # Compute the log of D1/D2 (except on the diagonal)
    div = torch.log(D_1[idx]/D_2[idx])

    # Return the Hilbert projective metric.
    return float((div.max() - div.min()).cpu())

def normalize_dataset(
    dataset: torch.Tensor, normalization_steps: int = 1,
    small_value: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
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
    assert(normalization_steps > 0)

    # Do a first normalization pass for A
    A = (small_value + dataset)
    A = A/A.sum(0)

    # Do a first normalization pass for B
    B = (small_value + dataset).T
    B = B/B.sum(0)

    # Make any additional normalization steps.
    for _ in range(normalization_steps - 1):
        A, B = B.T/B.T.sum(0), A.T/A.T.sum(0)
    
    return A, B


################################ DISTANCE MAPS ################################

def wasserstein_map(
    A: torch.Tensor, C: torch.Tensor, R: torch.Tensor,
    tau: float, dtype: str, device: str) -> torch.Tensor:
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

    # Iterate over the lines.
    for i in range(1, A.shape[1]):

        # Compute the Wasserstein distances.
        wass = ot.emd2(A[:,i].contiguous(), A[:,:i].contiguous(), C)

        # Add them in the distance matrix (including symmetric values).
        D[i,:i] = D[:i,i] = torch.Tensor(wass)
    
    # If the regularization parameter is > 0, regularize.
    if tau > 0:
        D = D + tau*R
    
    # Return the distance matrix.
    return D

def sinkhorn_map(A: torch.Tensor, C: torch.Tensor, R: torch.Tensor,
    tau: float, eps: float, dtype: str, device: str) -> torch.Tensor:
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

    # Iterate over the lines.
    for i in range(A.shape[1]):

        # Compute the Sinkhorn costs
        # TODO: this only returns the linear term. Use logs
        wass = ot.sinkhorn(A[:,i].contiguous(), A[:,:i+1].contiguous(), C, eps)

        # Add them in the distance matrix (including symmetric values).
        D[i,:i+1] = D[:i+1,i] = torch.Tensor(wass)
    
    # Get the diagonal terms OT_eps(a, a)
    d = torch.diagonal(D)

    # Sinkhorn divergence OT(a, b) - (OT(a, a) + OT(b, b))/2
    D = D - .5*(d.view(-1, 1) + d.view(-1, 1))

    # Make sure there are no negative values.
    assert((D < 0).sum() == 0)

    # Make sure the diagonal is zero.
    D.fill_diagonal_(0)
    
    # If the regularization parameter is > 0, regularize.
    if tau > 0:
        D = D + tau*R
    
    # Return the distance matrix.
    return D

############################### STOCHASTIC MAPS ###############################

def stochastic_wasserstein_map(
    A: torch.Tensor, D: torch.Tensor, C: torch.Tensor, R: torch.Tensor,
    sample_prop: float, tau: float, dtype: str, device: str) -> torch.Tensor:
    """Returns the stochastic Wasserstein map, updating only a random subset of
    indices and leaving the other ones as they are.

    Args:
        A (torch.Tensor): The input dataset.
        D (torch.Tensor): The intialization of the distance matrix
        C (torch.Tensor): The ground cost
        R (torch.Tensor): The regularization matrix.
        sample_size (int): The number of indices to update (they are symmetric)
        tau (float): The regularizatino parameter for R
        dtype (str): The dtype
        device (str): The device

    Returns:
        torch.Tensor: The stochastically updated distance matrix.
    """    
    
    # Name the dimensions of the dataset (features x samples).
    m, n = A.shape

    # Define the sample size from the proportion.
    sample_size = n*(n-1)/2
    sample_size = max(1, int(sample_prop*sample_size))

    # The indices to sample from
    # TODO: is this the right call ?
    ii, jj = torch.tril_indices(n, n, offset=-1)

    # The random indices to update.
    kk = np.random.choice(range(len(ii)), size=sample_size, replace=False)

    # Get the smallest nonzero value of R.
    #r = R[R > 0].min()
    r = R[ii[kk], jj[kk]]
    r = r[r > 0].min()

    # Initialize new distance
    D_new = D.clone()

    # Iterate over random indices.
    for k in kk:

        # Define the index to update.
        i, j = ii[k], jj[k]

        # Compute the Wasserstein distances.
        wass = ot.emd2(A[:,i].contiguous(), A[:,j].contiguous(), C)

        # Add them in the distance matrix (including symmetric values).
        # Also add regularization.
        # TODO: is this inplace ?
        D_new[i,j] = D_new[j,i] = (wass + tau*R[i,j])/r#*tau)
    
    # Return the distance matrix.
    return D_new

def stochastic_sinkhorn_map(
    A: torch.Tensor, D: torch.Tensor, C: torch.Tensor,
    R: torch.Tensor, sample_prop: float, tau: float,
    eps: float, dtype: str, device: str) -> torch.Tensor:
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
    sample_size = max(2, int(sample_prop*sample_size))

    # Random indices.
    ii = np.random.choice(range(n), size=sample_size, replace=False)

    # Get the smallest nonzero value of R.
    #r = R[R > 0].min()
    r = R[ii][:,ii]
    r = r[r > 0].min()

    # Initialize new distance
    D_new = D.clone()

    # Iterate over random indices.
    # TODO: is this the way?
    # Iterate over the lines.
    for k in range(sample_size):

        # Compute the Sinkhorn costs
        # TODO: this only returns the linear term. Use logs
        wass = ot.sinkhorn(A[:,ii[k]].contiguous(), A[:,ii[:k+1]].contiguous(), C, eps)

        # Add them in the distance matrix (including symmetric values).
        D_new[ii[k],ii[:k+1]] = D_new[ii[:k+1],ii[k]] = torch.Tensor(wass)
    
    # Get the diagonal terms OT_eps(a, a)
    d = torch.diagonal(D_new)

    # Sinkhorn divergence OT(a, b) - (OT(a, a) + OT(b, b))/2
    D_new = D_new - .5*(d.view(-1, 1) + d.view(-1, 1))

    # Make sure there are no negative values.
    assert((D_new < 0).sum() == 0)

    # Make sure the diagonal is zero.
    D_new.fill_diagonal_(0)

    xx, yy = np.meshgrid(ii, ii)

    # If the regularization parameter is > 0, regularize.
    if tau > 0:
        D_new[xx, yy] += tau*R[xx, yy]
    
    r = R[xx, yy]
    r = r[r > 0].min()
    D_new[xx, yy] /= r
    
    # Return the distance matrix.
    return D_new

############################# THE POWER ITERATIONS ############################

def wasserstein_singular_vectors(
    dataset: torch.Tensor, tau: float,
    p: int, dtype: str, device: str, max_iter: int,
    writer: SummaryWriter, small_value: float = 1e-6,
    normalization_steps: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
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
    A, B = normalize_dataset(dataset, normalization_steps, small_value)
    
    # Initialize a random cost matrix.
    C = random_distance(m, dtype=dtype, device=device)
    D = random_distance(n, dtype=dtype, device=device)

    # Compute the regularization matrices.
    R_A = regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = regularization_matrix(B, p=p, dtype=dtype, device=device)

    # Iterate until `max_iter`.
    for n_iter in range(max_iter):

        try:
            # Compute D using C
            D_new = wasserstein_map(A, C, R_A, tau=tau, dtype=dtype, device=device)

            # Compute Hilbert loss
            if writer:
                writer.add_scalar('Hilbert D', hilbert_distance(D, D_new), n_iter)

            # Normalize D
            D = D_new/D_new.max()

            # Compute C using D
            C_new = wasserstein_map(B, D, R_B, tau=tau, dtype=dtype, device=device)

            # Compute Hilbert loss
            if writer:
                writer.add_scalar('Hilbert C', hilbert_distance(C, C_new), n_iter)

            # Normalize C
            C = C_new/C_new.max()

            # TODO: Try early stopping.
        
        except KeyboardInterrupt:
            print('Stopping early after keyboard interrupt!')
            C /= C.max()
            D /= D.max()
            break

    return C, D

def sinkhorn_singular_vectors(
    dataset: torch.Tensor, tau: float, eps: float, p: int,
    dtype: str, device: str, max_iter: int,
    writer: SummaryWriter, small_value: float = 1e-6,
    normalization_steps: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
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
    A, B = normalize_dataset(dataset, normalization_steps, small_value)
    
    # Initialize a random cost matrix.
    C = random_distance(m, dtype=dtype, device=device)
    D = random_distance(n, dtype=dtype, device=device)    

    # Compute the regularization matrices.
    R_A = regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = regularization_matrix(B, p=p, dtype=dtype, device=device)

    # Iterate until `max_iter`.
    for n_iter in range(max_iter):

        try:

            # Compute D using C
            D_new = sinkhorn_map(
                A, C, R_A, tau=tau, eps=eps,
                dtype=dtype, device=device)

            # Compute Hilbert loss
            if writer:
                writer.add_scalar('Hilbert D', hilbert_distance(D, D_new), n_iter)

            # Normalize D
            D = D_new/D_new.max()

            # Compute C using D
            C_new = sinkhorn_map(
                B, D, R_B, tau=tau, eps=eps,
                dtype=dtype, device=device)

            # Compute Hilbert loss
            if writer:
                writer.add_scalar('Hilbert C', hilbert_distance(C, C_new), n_iter)

            # Normalize C
            C = C_new/C_new.max()

            # TODO: Try early stopping.
        
        except KeyboardInterrupt:
            print('Stopping early after keyboard interrupt!')
            C /= C.max()
            D /= D.max()
            break

    return C, D

####################### THE STOCHASTIC POWER ITERATIONS #######################

def stochastic_wasserstein_singular_vectors(
    dataset: torch.Tensor, tau: float, sample_prop: float,
    p: int, dtype: str, device: str, max_iter: int,
    writer: SummaryWriter, small_value: float = 1e-6,
    normalization_steps: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
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
    A, B = normalize_dataset(dataset, normalization_steps, small_value)
    
    # Initialize a random cost matrix.
    C = random_distance(m, dtype=dtype, device=device)
    D = random_distance(n, dtype=dtype, device=device)

    # Compute the regularization matrices.
    R_A = regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = regularization_matrix(B, p=p, dtype=dtype, device=device)

    # Iterate until `max_iter`.
    for k in range(1, max_iter):

        try:

            # Ste the decreasing step size.
            step_size = 1/np.sqrt(k)
            writer.add_scalar('step_size', step_size, k)

            # Compute D using C
            D_new = (1 - step_size)*D + step_size*stochastic_wasserstein_map(
                A, D, C, R_A, sample_prop=sample_prop,
                tau=tau, dtype=dtype, device=device)

            # Compute Hilbert loss.
            if writer:
                writer.add_scalar('Hilbert D', hilbert_distance(D, D_new), k)

            # Normalize D.
            D = D_new/D_new.max()

            # Compute C using D
            C_new = (1 - step_size)*C + step_size*stochastic_wasserstein_map(
                B, C, D, R_B, sample_prop=sample_prop,
                tau=tau, dtype=dtype, device=device)

            # Compute Hilbert loss.
            if writer:
                writer.add_scalar('Hilbert C', hilbert_distance(C, C_new), k)

            # Normalize D.
            C = C_new/C_new.max()

            # TODO: Try early stopping.
        
        except KeyboardInterrupt:
            print('Stopping early after keyboard interrupt!')
            C /= C.max()
            D /= D.max()
            break

    return C, D

def stochastic_sinkhorn_singular_vectors(
    dataset: torch.Tensor, tau: float, eps: float, sample_prop: float,
    p: int, dtype: str, device: str, max_iter: int,
    writer: SummaryWriter, small_value: float = 1e-6,
    normalization_steps: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
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
    A, B = normalize_dataset(dataset, normalization_steps, small_value)
    
    # Initialize a random cost matrix.
    C = random_distance(m, dtype=dtype, device=device)
    D = random_distance(n, dtype=dtype, device=device)

    # Compute the regularization matrices.
    R_A = regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = regularization_matrix(B, p=p, dtype=dtype, device=device)

    # Iterate until `max_iter`.
    for k in range(1, max_iter):

        try:

            # Set the decresing step size.
            step_size = 1/np.sqrt(k)
            writer.add_scalar('step_size', step_size, k)

            # Compute D using C
            D_new = (1 - step_size)*D + step_size*stochastic_sinkhorn_map(
                A, D, C, R_A, sample_prop=sample_prop, tau=tau,
                eps=eps, dtype=dtype, device=device)

            # Compute Hilbert loss.
            writer.add_scalar('Hilbert D', hilbert_distance(D, D_new), k)

            # Normalize D.
            D = D_new/D_new.max()

            # Compute C using D
            C_new = (1 - step_size)*C + step_size*stochastic_sinkhorn_map(
                B, C, D, R_B, sample_prop=sample_prop, tau=tau,
                eps=eps, dtype=dtype, device=device)

            # Compute Hilbert loss.
            writer.add_scalar('Hilbert C', hilbert_distance(C, C_new), k)

            # Normalize D.
            C = C_new/C_new.max()

            # TODO: Try early stopping.
        
        except KeyboardInterrupt:
            print('Stopping early after keyboard interrupt!')
            C /= C.max()
            D /= D.max()
            break

    return C, D