################################### IMPORTS ###################################

# Matrices
import torch
import numpy as np
import pandas as pd

# Optimal Transport
import ot

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Typing
from typing import Callable, Iterable, Tuple

# Silhouette score
from sklearn.metrics import silhouette_score

# TSNE
from sklearn.manifold import TSNE

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Progress bar
from tqdm import tqdm

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
    if p=='one':
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
    idx = (torch.eye(D_1.shape[0]) != 1)

    # Compute the log of D1/D2 (except on the diagonal)
    div = torch.log(D_1[idx]/D_2[idx])

    # Return the Hilbert projective metric.
    return float((div.max() - div.min()).cpu())

def normalize_dataset(
    dataset: torch.Tensor, dtype: str, device: str,
    normalization_steps: int = 1, small_value: float = 1e-6,
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
    assert(normalization_steps > 0)

    # Do a first normalization pass for A
    A = dataset/dataset.sum(0)
    A += small_value
    A /= A.sum(0)

    # Do a first normalization pass for B
    B = dataset.T/dataset.T.sum(0)
    B += small_value
    B /= B.sum(0)

    # Make any additional normalization steps.
    for _ in range(normalization_steps - 1):
        A, B = B.T/B.T.sum(0), A.T/A.T.sum(0)
    
    return A.to(dtype=dtype, device=device), B.to(dtype=dtype, device=device)

def check_uniqueness(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
    D: torch.Tensor, dtype: str, device: str) -> bool:
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
    return silhouette_score(D.cpu(), labels, metric='precomputed')

def knn_error(D: torch.Tensor, k: int, labels_train: Iterable, labels_test: Iterable) -> float:
    """Returns the KNN error.

    Args:
        D (torch.Tensor): Input distance matrix
        k (int): How many neighbours
        labels (Iterable): The labels

    Returns:
        float: The KNN error
    """

    # Get k+1 lowest distances.
    # _, indices = D.topk(k=k+1, dim=1, largest=False)

    # # Remove the index itself (it has distance 0 of course).
    # different_indices = []
    # for i in range(indices.shape[0]):
    #     idx = (indices[i] != i)
    #     different_indices.append(indices[i, idx])

    # error = []
    # for i in range(len(labels)):
    #     label_array = np.array(labels)[different_indices[i]]
    #     error.append(np.mean(label_array != np.array(labels)[i]))
    
    # return np.mean(error), different_indices

    label_train_codes = pd.Categorical(labels_train).codes
    label_test_codes = pd.Categorical(labels_test).codes

    acc = 0
    for i in range(label_test_codes.shape[0]):
        rank = np.argsort(D[i])
        if np.bincount(label_train_codes[rank[:k]]).argmax() == label_test_codes[i]:
            acc += 1
    
    acc = acc / label_test_codes.shape[0]
    return 1 - acc



def viz_TSNE(
    D: torch.Tensor, labels: Iterable = None,
    names: Iterable = [], save_path: str = None, p=.1) -> None:
    """Visualize a distance matrix using a precomputed distance matrix.

    Args:
        D (torch.Tensor): Distance matrix
        labels (Iterable, optional): The labels, if any. Defaults to None.
    """    
    tsne = TSNE(
        n_components=2, random_state=0,
        metric='precomputed', square_distances=True)
    embed = tsne.fit_transform(D.cpu())
    df = pd.DataFrame(embed, columns=['x', 'y'])
    df['label'] = labels
    sns.scatterplot(data=df, x='x', y='y', hue='label')
    if len(names) > 0:
        for i in range(df.shape[0]):
            if np.random.choice([True, False], p=(p, 1-p)):
                plt.text(x=df.x[i]+0.3,y=df.y[i]+0.3,s=names[i])
    if save_path:
        plt.savefig(save_path)
    plt.close()

################################ DISTANCE MAPS ################################

def wasserstein_map(
    A: torch.Tensor, C: torch.Tensor, R: torch.Tensor,
    tau: float, dtype: str, device: str, progress_bar=False) -> torch.Tensor:
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
        pbar = tqdm(total=A.shape[1]*(A.shape[1] - 1)//2, leave=False)

    # Iterate over the lines.
    for i in range(1, A.shape[1]):

        if progress_bar:
            pbar.update(i)

        # Compute the Wasserstein distances.
        wass = ot.emd2(A[:,i].contiguous(), A[:,:i].contiguous(), C)

        # Add them in the distance matrix (including symmetric values).
        D[i,:i] = D[:i,i] = torch.Tensor(wass)
    
    if progress_bar:
        pbar.close()
    
    # If the regularization parameter is > 0, regularize.
    if tau > 0:
        D = D + tau*R
    
    # Return the distance matrix.
    return D

def sinkhorn_map(A: torch.Tensor, C: torch.Tensor, R: torch.Tensor,
    tau: float, eps: float, dtype: str, device: str, progress_bar=False) -> torch.Tensor:
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

    K = (-C/eps).exp()

    if progress_bar:
        pbar = tqdm(total=A.shape[1]*(A.shape[1] - 1)//2, leave=False)

    # Iterate over the lines.
    for i in range(A.shape[1]):

        for ii in np.array_split(range(i+1), max(1, i//100)):

            # Compute the Sinkhorn dual variables
            _, wass_log = ot.sinkhorn(
                A[:,i].contiguous(), # This is the source histogram.
                A[:,ii].contiguous(), # These are the target histograms.
                C, # This is the ground cost.
                eps, # This is the regularization parameter.
                log=True, # Return the dual variables
                stopThr=1e-5,
                numItermax=500
            )

            # Compute the exponential dual potentials.
            f, g = eps*wass_log['u'].log(), eps*wass_log['v'].log()

            # Compute the Sinkhorn costs.
            # These will be used to compute the Sinkhorn divergences
            wass = (
                f*A[:,[i]*len(ii)] +
                g*A[:,ii] -
                eps*wass_log['u']*(K@wass_log['v'])
            ).sum(0)

            # Add them in the distance matrix (including symmetric values).
            D[i,ii] = D[ii,i] = wass

            if progress_bar:
                pbar.update(len(ii))
    
    if progress_bar:
            pbar.close()
    
    # Get the diagonal terms OT_eps(a, a).
    d = torch.diagonal(D)

    # The Sinkhorn divergence is OT(a, b) - (OT(a, a) + OT(b, b))/2.
    D = D - .5*(d.view(-1, 1) + d.view(1, -1))

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
    A: torch.Tensor, D: torch.Tensor, C: torch.Tensor,
    R: torch.Tensor, sample_prop: float, tau: float,
    gamma: float, dtype: str, device: str,
    progress_bar=False, return_indices=False) -> torch.Tensor:
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

    assert(gamma > 0)
    assert(tau >= 0)
    #TODO assert simplex
    
    # Name the dimensions of the dataset (features x samples).
    m, n = A.shape

    # Define the sample size from the proportion. TODO: Not a linear function though
    sample_size = n
    sample_size = max(2, int(np.sqrt(sample_prop)*sample_size))

    # The indices to sample from
    # Random indices.
    ii = np.random.choice(range(n), size=sample_size, replace=False)

    # Initialize new distance
    D_new = D.clone()

    if progress_bar:
        pbar = tqdm(total=sample_size*(sample_size - 1)//2, leave=False)

    # Iterate over random indices.
    for k in range(1, sample_size):

        if progress_bar:
            pbar.update(k)

        # Compute the Wasserstein distances.
        wass = torch.Tensor(ot.emd2(A[:,ii[k]].contiguous(), A[:,ii[:k]].contiguous(), C)).to(dtype=dtype, device=device)

        # Add them in the distance matrix (including symmetric values).
        # Also add regularization.
        D_new[ii[k],ii[:k]] = D_new[ii[:k],ii[k]] = wass
    
    if progress_bar:
        pbar.close()
    
    # Make sure the diagonal is zero.
    D_new.fill_diagonal_(0)

    # Get the indices for the grid (ii,ii).
    xx, yy = np.meshgrid(ii, ii)

    # If the regularization parameter is > 0, regularize.
    if tau > 0:
        D_new[xx, yy] += tau*R[xx, yy]
    
    # Divide gamma
    D_new[xx, yy] /= gamma
    
    # Return the distance matrix.
    if return_indices:
        return D_new, xx, yy
    else:
        return D_new


def stochastic_sinkhorn_map(
    A: torch.Tensor, D: torch.Tensor, C: torch.Tensor,
    R: torch.Tensor, sample_prop: float, tau: float, gamma: float,
    eps: float, dtype: str, device: str, progress_bar=False,
    return_indices=False, batch_size=50) -> torch.Tensor:
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
    sample_size = max(2, int(np.sqrt(sample_prop)*sample_size))

    # Random indices.
    idx = np.random.choice(range(n), size=sample_size, replace=False)

    # Initialize new distance
    D_new = D.clone()

    K = (-C/eps).exp()

    if progress_bar:
        pbar = tqdm(total=sample_size*(sample_size - 1)//2, leave=False)

    # Iterate over random indices.
    for k in range(sample_size):

        i = idx[k]
        # ii = idx[:k+1]

        for ii in np.array_split(idx[:k+1], max(1, k//batch_size)):

            # Compute the Sinkhorn dual variables.
            _, wass_log = ot.sinkhorn(
                A[:,i].contiguous(), # This is the source histogram.
                A[:,ii].contiguous(), # These are the target histograms.
                C, # This is the gruond cost.
                eps, # This is the entropic regularization parameter.
                log=True, # Return the dual variables.
                stopThr=1e-5,
                numItermax=100
            )

            # Compute the exponential dual variables.
            f, g = eps*wass_log['u'].log(), eps*wass_log['v'].log()

            # Compute the Sinkhorn costs.
            # These will be used to compute the Sinkhorn divergences below.
            wass = (
                f*A[:,[i]*len(ii)] +
                g*A[:,ii] -
                eps*wass_log['u']*(K@wass_log['v'])
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
    D_new[xx, yy] = D_new[xx, yy] - .5*(d.view(-1, 1) + d.view(1, -1))

    # Make sure there are no negative values.
    # assert((D_new < 0).sum() == 0)

    # Make sure the diagonal is zero.
    D_new[xx, yy].fill_diagonal_(0)


    # If the regularization parameter is > 0, regularize.
    if tau > 0:
        D_new[xx, yy] += tau*R[xx, yy]
    
    # Divide gamma
    D_new[xx, yy] /= gamma
    
    # Return the distance matrix.
    if return_indices:
        return D_new, xx, yy
    else:
        return D_new

############################# THE POWER ITERATIONS ############################

# TODO: implement reference C and D for log

def wasserstein_singular_vectors(
    dataset: torch.Tensor, tau: float,
    p: int, dtype: str, device: str, max_iter: int,
    writer: SummaryWriter, small_value: float = 1e-6,
    normalization_steps: int = 1, C_ref: torch.tensor = None,
    D_ref: torch.Tensor = None, log_loss=False, progress_bar=False
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
    A, B = normalize_dataset(
        dataset,
        normalization_steps=normalization_steps,
        small_value=small_value,
        dtype=dtype, device=device)
    
    # Initialize a random cost matrix.
    C = random_distance(m, dtype=dtype, device=device)
    D = random_distance(n, dtype=dtype, device=device)

    # Compute the regularization matrices.
    R_A = regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = regularization_matrix(B, p=p, dtype=dtype, device=device)

    # Initialize loss history.
    loss_C, loss_D = [], []

    # Iterate until `max_iter`.
    for n_iter in range(max_iter):

        try:
            # Compute D using C
            D_new = wasserstein_map(
                A, C, R_A, tau=tau, dtype=dtype,
                device=device, progress_bar=progress_bar)

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(D_ref):
                    loss_D.append(hilbert_distance(D, D_ref))
                    writer.add_scalar('Hilbert D,D_ref', loss_D[-1], n_iter)
                writer.add_scalar('Hilbert D,D_new', hilbert_distance(D, D_new), n_iter)

            # Normalize D
            D = D_new/D_new.max()

            # Compute C using D
            C_new = wasserstein_map(B, D, R_B, tau=tau, dtype=dtype,
                device=device, progress_bar=progress_bar)

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(C_ref):
                    loss_C.append(hilbert_distance(C, C_ref))
                    writer.add_scalar('Hilbert C,C_ref', loss_C[-1], n_iter)
                writer.add_scalar('Hilbert C,C_new', hilbert_distance(C, C_new), n_iter)

            # Normalize C
            C = C_new/C_new.max()

            # TODO: Try early stopping.
        
        except KeyboardInterrupt:
            print('Stopping early after keyboard interrupt!')
            C /= C.max()
            D /= D.max()
            break
        
    if log_loss:
        return C, D, loss_C, loss_D
    else:
        return C, D

def sinkhorn_singular_vectors(
    dataset: torch.Tensor, tau: float, eps: float, p: int,
    dtype: str, device: str, max_iter: int,
    writer: SummaryWriter, small_value: float = 1e-6,
    normalization_steps: int = 1, C_ref: torch.tensor = None,
    D_ref: torch.Tensor = None, log_loss=False,
    progress_bar=False) -> Tuple[torch.Tensor, torch.Tensor]:
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
    A, B = normalize_dataset(
        dataset,
        normalization_steps=normalization_steps,
        small_value=small_value,
        dtype=dtype, device=device)
    
    # Initialize a random cost matrix.
    C = random_distance(m, dtype=dtype, device=device)
    D = random_distance(n, dtype=dtype, device=device)

    # Compute the regularization matrices.
    R_A = regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = regularization_matrix(B, p=p, dtype=dtype, device=device)

    # Initialize loss history.
    loss_C, loss_D = [], []

    # Iterate until `max_iter`.
    for n_iter in range(max_iter):

        try:

            # Compute D using C
            D_new = sinkhorn_map(
                A, C, R_A, tau=tau, eps=eps,
                dtype=dtype, device=device, progress_bar=progress_bar)

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(D_ref):
                    loss_D.append(hilbert_distance(D, D_ref))
                    writer.add_scalar('Hilbert D,D_ref', loss_D[-1], n_iter)
                writer.add_scalar('Hilbert D,D_new', hilbert_distance(D, D_new), n_iter)

            # Normalize D
            D = D_new/D_new.max()

            # Compute C using D
            C_new = sinkhorn_map(
                B, D, R_B, tau=tau, eps=eps,
                dtype=dtype, device=device, progress_bar=progress_bar)

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(C_ref):
                    loss_C.append(hilbert_distance(C, C_ref))
                    writer.add_scalar('Hilbert C,C_ref', loss_C[-1], n_iter)
                writer.add_scalar('Hilbert C,C_new', hilbert_distance(C, C_new), n_iter)

            # Normalize C
            C = C_new/C_new.max()

            # TODO: Try early stopping.
        
        except KeyboardInterrupt:
            print('Stopping early after keyboard interrupt!')
            C /= C.max()
            D /= D.max()
            break
        
    if log_loss:
        return C, D, loss_C, loss_D
    else:
        return C, D

####################### THE STOCHASTIC POWER ITERATIONS #######################

def stochastic_wasserstein_singular_vectors(
    dataset: torch.Tensor, tau: float, sample_prop: float,
    p: int, dtype: str, device: str, max_iter: int,
    writer: SummaryWriter, small_value: float = 1e-6,
    normalization_steps: int = 1, C_ref: torch.tensor = None,
    D_ref: torch.Tensor = None, progress_bar=False,
    step_fn: Callable = lambda k:1/np.sqrt(k), mult_update=False) -> Tuple[torch.Tensor, torch.Tensor]:
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
    A, B = normalize_dataset(
        dataset,
        normalization_steps=normalization_steps,
        small_value=small_value,
        dtype=dtype, device=device)

    # Compute the regularization matrices.
    R_A = regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = regularization_matrix(B, p=p, dtype=dtype, device=device)

    D = R_A.clone()
    C = R_B.clone()

    mu, lbda = 1, 1

    # Iterate until `max_iter`.
    for k in range(1, max_iter):

        try:

            # Set the decreasing step size.
            step_size = step_fn(k)
            writer.add_scalar('step_size', step_size, k)

            C_new, xx, yy = stochastic_wasserstein_map(B, C, D, gamma=1, sample_prop=sample_prop, R=R_B, tau=tau, dtype=dtype, device=device, return_indices=True)
    
            lbda = (1-step_size)*lbda + step_size*torch.sum(C_new[xx, yy]*C[xx, yy])/torch.sum(C[xx, yy]**2)

            if writer:
                writer.add_scalar('lambda', lbda, k)

            C_new[xx, yy] /= lbda

            if mult_update:
                C_new = torch.exp((1-step_size)*C.log() + step_size*C_new.log())
            else:
                C_new = (1-step_size)*C + step_size*C_new
            
            C_new.fill_diagonal_(0)

            if writer:
                if torch.is_tensor(C_ref):
                    hilbert = hilbert_distance(C_new, C_ref)    
                    writer.add_scalar('Hilbert C,C_ref', hilbert, k)
                writer.add_scalar('Hilbert C,C_new', hilbert_distance(C, C_new), k)
            
            C = C_new/C_new.max()
            
            D_new, xx, yy = stochastic_wasserstein_map(A, D, C, gamma=1, sample_prop=sample_prop, R=R_A, tau=tau, dtype=dtype, device=device, return_indices=True)
            
            mu = (1-step_size)*mu + step_size*torch.sum(D_new[xx, yy]*D[xx, yy])/torch.sum(D[xx, yy]**2)

            if writer:
                writer.add_scalar('mu', mu, k)

            D_new[xx, yy] /= mu

            if mult_update:
                D_new = torch.exp((1-step_size)*D.log() + step_size*D_new.log())
            else:
                D_new = (1-step_size)*D + step_size*D_new

            D_new.fill_diagonal_(0)

            if writer:
                if torch.is_tensor(D_ref):
                    hilbert = hilbert_distance(D_new, D_ref)
                    writer.add_scalar('Hilbert D,D_ref', hilbert, k)
                writer.add_scalar('Hilbert D,D_new', hilbert_distance(D, D_new), k)
            
            D = D_new/D_new.max()
        
        except KeyboardInterrupt:
            print('Stopping early after keyboard interrupt!')
            C /= C.max()
            D /= D.max()
            break
    return C, D

def stochastic_sinkhorn_singular_vectors(
    dataset: torch.Tensor, tau: float, eps: float,
    sample_prop: float, p: int, dtype: str, device: str, max_iter: int,
    writer: SummaryWriter, small_value: float = 1e-6, C_ref=None, D_ref=None,
    normalization_steps: int = 1, step_fn: Callable = lambda k:2/(2+np.sqrt(k)),
    progress_bar=False, mult_update=False) -> Tuple[torch.Tensor, torch.Tensor]:
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
    A, B = normalize_dataset(
        dataset,
        normalization_steps=normalization_steps,
        small_value=small_value,
        dtype=dtype, device=device)
    
    # Initialize a random cost matrix.
    # C = random_distance(m, dtype=dtype, device=device)
    # D = random_distance(n, dtype=dtype, device=device)

    # Compute the regularization matrices.
    R_A = regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = regularization_matrix(B, p=p, dtype=dtype, device=device)

    D = R_A.clone()
    C = R_B.clone()

    mu, lbda = 1, 1

    # Iterate until `max_iter`.
    for k in range(1, max_iter):

        try:

            # Set the decreasing step size.
            step_size = step_fn(k)
            writer.add_scalar('step_size', step_size, k)

            C_new, xx, yy = stochastic_sinkhorn_map(B, C, D, gamma=1, sample_prop=sample_prop, R=R_B, tau=tau, eps=eps, dtype=dtype, device=device, return_indices=True, progress_bar=progress_bar)
    
            lbda = (1-step_size)*lbda + step_size*torch.sum(C_new[xx, yy]*C[xx, yy])/torch.sum(C[xx, yy]**2)

            if writer:
                writer.add_scalar('lambda', lbda, k)

            C_new[xx, yy] /= lbda

            if mult_update:
                C_new = torch.exp((1-step_size)*C.log() + step_size*C_new.log())
            else:
                C_new = (1-step_size)*C + step_size*C_new
            
            C_new.fill_diagonal_(0)

            if writer:
                if torch.is_tensor(C_ref):
                    hilbert = hilbert_distance(C_new, C_ref)    
                    writer.add_scalar('Hilbert C,C_ref', hilbert, k)
            writer.add_scalar('Hilbert C,C_new', hilbert_distance(C, C_new), k)
            
            C = C_new/C_new.max()
            
            D_new, xx, yy = stochastic_sinkhorn_map(A, D, C, gamma=1, sample_prop=sample_prop, R=R_A, tau=tau, eps=eps, dtype=dtype, device=device, return_indices=True, progress_bar=progress_bar)
            
            mu = (1-step_size)*mu + step_size*torch.sum(D_new[xx, yy]*D[xx, yy])/torch.sum(D[xx, yy]**2)

            if writer:
                writer.add_scalar('mu', mu, k)

            D_new[xx, yy] /= mu

            if mult_update:
                D_new = torch.exp((1-step_size)*D.log() + step_size*D_new.log())
            else:
                D_new = (1-step_size)*D + step_size*D_new

            D_new.fill_diagonal_(0)

            if writer:
                if torch.is_tensor(D_ref):
                    hilbert = hilbert_distance(D_new, D_ref)
                    writer.add_scalar('Hilbert D,D_ref', hilbert, k)
            writer.add_scalar('Hilbert D,D_new', hilbert_distance(D, D_new), k)
            
            D = D_new/D_new.max()
        
        except KeyboardInterrupt:
            print('Stopping early after keyboard interrupt!')
            C /= C.max()
            D /= D.max()
            break
        
    return C, D