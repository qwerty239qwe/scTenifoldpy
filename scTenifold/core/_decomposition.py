from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scTenifold.core._rng import RRandom
from scTenifold.core._utils import timer

__all__ = ["cp_decomposition", "tensor_decomp"]


def cp_decomposition(tensor: np.ndarray,
                     K: int,
                     max_iter: int = 1000,
                     tol: float = 1e-5,
                     random_state: Optional[int] = 1) -> dict:
    """CANDECOMP/PARAFAC decomposition of a 3-mode tensor by alternating least squares.

    A port of ``cpDecomposition`` in the R package scTenifoldNet: the factors
    are initialized with R's ``rnorm`` after ``set.seed(random_state)``, each
    column is normalized by its L1 norm, and the iteration stops when the
    change of the residual Frobenius norm, relative to the norm of the
    tensor, is below ``tol``, or after ``max_iter - 1`` iterations.

    Parameters
    ----------
    tensor
        Tensor of shape ``(I, J, K_slices)``.
    K
        Number of rank-one components.
    max_iter
        Maximum number of iterations.
    tol
        Relative Frobenius norm error tolerance.
    random_state
        Seed of the initial factors, as ``set.seed`` in R.

    Returns
    -------
    Dictionary with ``lambdas``, the factor matrices ``U``, whether it
    converged (``conv``), the residual norm of each iteration
    (``all_resids``), the percent of the norm explained (``norm_percent``)
    and ``slice_sum``, the sum over the last mode of the estimated tensor.
    """
    tensor = np.asarray(tensor, dtype=float)
    if tensor.ndim != 3:
        raise ValueError("cp_decomposition expects a 3-mode tensor")
    I, J, n_slices = tensor.shape
    tnsr_norm_sq = np.sum(tensor * tensor)
    tnsr_norm = np.sqrt(tnsr_norm_sq)

    # R fills each factor column by column from one rnorm() call per mode
    rng = RRandom(random_state)
    U = [rng.rnorm(m * K).reshape((m, K), order="F") for m in (I, J, n_slices)]
    # tensor[:, :, k] is strided; contiguous copies make the products much faster
    slices = [np.ascontiguousarray(tensor[:, :, k]) for k in range(n_slices)]

    curr_iter = 1
    converged = False
    resids = []
    lambdas = np.zeros(K)
    prev_resid = np.inf
    while curr_iter < max_iter and not converged:
        # Slicewise MTTKRP, avoiding the Khatri-Rao products
        V = (U[1].T @ U[1]) * (U[2].T @ U[2])
        mttkrp = np.zeros((I, K))
        for k in range(n_slices):
            mttkrp = mttkrp + (slices[k] @ U[1]) * U[2][k, :]
        tmp = mttkrp @ np.linalg.inv(V)
        lambdas = np.abs(tmp).sum(axis=0)
        U[0] = tmp / lambdas

        V = (U[0].T @ U[0]) * (U[2].T @ U[2])
        mttkrp = np.zeros((J, K))
        for k in range(n_slices):
            mttkrp = mttkrp + (slices[k].T @ U[0]) * U[2][k, :]
        tmp = mttkrp @ np.linalg.inv(V)
        lambdas = np.abs(tmp).sum(axis=0)
        U[1] = tmp / lambdas

        V = (U[0].T @ U[0]) * (U[1].T @ U[1])
        mttkrp = np.zeros((n_slices, K))
        for k in range(n_slices):
            mttkrp[k, :] = (U[0] * (slices[k] @ U[1])).sum(axis=0)
        tmp = mttkrp @ np.linalg.inv(V)
        lambdas = np.abs(tmp).sum(axis=0)
        U[2] = tmp / lambdas

        # ||X - est||^2 = ||X||^2 - 2 <X, est> + ||est||^2, without
        # reconstructing the tensor
        inner = np.sum(lambdas * (U[2] * mttkrp).sum(axis=0))
        gamma = (U[0].T @ U[0]) * (U[1].T @ U[1]) * (U[2].T @ U[2])
        est_norm_sq = np.sum(np.outer(lambdas, lambdas) * gamma)
        curr_resid = np.sqrt(max(tnsr_norm_sq - 2 * inner + est_norm_sq, 0))

        resids.append(curr_resid)
        if curr_iter > 1 and abs(curr_resid - prev_resid) / tnsr_norm < tol:
            converged = True
        else:
            prev_resid = curr_resid
            curr_iter += 1

    # Sum of the estimated slices, added one slice at a time as in R
    slice_sum = np.zeros((I, J))
    for k in range(n_slices):
        slice_sum = slice_sum + (U[0] * (lambdas * U[2][k, :])) @ U[1].T

    resids = [r for r in resids if r != 0]
    norm_percent = (1 - resids[-1] / tnsr_norm) * 100 if resids else np.nan
    return {"lambdas": lambdas, "U": U, "conv": converged, "all_resids": resids,
            "norm_percent": norm_percent, "slice_sum": slice_sum}


@timer
def tensor_decomp(networks: np.ndarray,
                  gene_names: Sequence[str],
                  method: str = "cp_als",
                  n_decimal: int = 1,
                  K: int = 5,
                  tol: float = 1e-5,
                  max_iter: int = 1000,
                  random_state: Optional[int] = 1,
                  **kwargs) -> pd.DataFrame:
    """
    Perform tensor decomposition on pc networks

    With ``method="cp_als"`` (default) this is ``tensorDecomposition`` of the
    R package scTenifoldNet, including its random initialization, so both
    give the same networks for the same seed.

    Parameters
    ----------
    networks: np.ndarray
        Concatenated network, expected shape = (n_genes, n_genes, n_pcnets)
    gene_names: sequence of str
        The name of each gene in the network (order matters)
    method: str, default = 'cp_als'
        ``"cp_als"`` for the CP decomposition of the R package, or the name of
        a tensorly decomposition method:
        http://tensorly.org/stable/modules/api.html#module-tensorly.decomposition
    n_decimal: int
        Number of decimal in the final df
    K: int
        Rank of the decomposition
    tol: float
        Tolerance in the iteration
    max_iter: int
        Number of interation
    random_state: int
        Random seed used to reproduce the same result
    **kwargs:
        Keyword arguments used in the tensorly decomposition function

    Returns
    -------
    tensor_decomp_df: pd.DataFrame
        The result of tensor decomposition, expected shape = (n_genes, n_genes)

    References
    ----------
    http://tensorly.org/stable/modules/api.html#module-tensorly.decomposition

    """
    if method == "cp_als":
        out = cp_decomposition(networks, K=K, max_iter=max_iter, tol=tol,
                               random_state=random_state)["slice_sum"]
    else:
        import tensorly as tl
        from tensorly import decomposition
        factors = getattr(decomposition, method)(networks, rank=K, n_iter_max=max_iter, tol=tol,
                                                 random_state=random_state, **kwargs)
        out = np.sum(tl.cp_to_tensor(factors), axis=-1)
    out = out / networks.shape[-1]
    out = np.round(out / np.max(abs(out)), n_decimal)
    return pd.DataFrame(out, index=gene_names, columns=gene_names)
