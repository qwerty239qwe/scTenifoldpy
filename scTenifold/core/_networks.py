from warnings import warn
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.linalg
from scipy import stats
from scipy.sparse import coo_matrix, issparse
import scipy.sparse.linalg
from sklearn.utils.extmath import randomized_svd

from scTenifold.core._rng import RRandom
from scTenifold.core._utils import cal_fdr, timer
from scTenifold.core._types import Backend, ExpressionData, LayerName


__all__ = ["make_networks", "cal_pcNet", "pc_net", "cal_pc_coefs", "manifold_alignment", "d_regulation", "strict_direction"]

NETWORK_BACKENDS = {"serial", "joblib-loky", "joblib-threading", "ray"}


def anndata_to_dataframe(data: ExpressionData, layer: LayerName = None) -> pd.DataFrame:
    """Convert a pandas or AnnData-like object to genes x cells DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    if not all(hasattr(data, attr) for attr in ("X", "var_names", "obs_names")):
        raise TypeError("data must be a pandas DataFrame or an AnnData-like object")
    matrix = data.X if layer is None else data.layers[layer]
    if issparse(matrix):
        matrix = matrix.toarray()
    return pd.DataFrame(np.asarray(matrix).T,
                        index=pd.Index(data.var_names),
                        columns=pd.Index(data.obs_names))


def _resolve_backend(backend: Backend, n_jobs: int, n_cpus: Optional[int]) -> tuple:
    if n_cpus is not None:
        warn("n_cpus is deprecated and will be removed in a future 0.2.x release; use n_jobs instead.",
             DeprecationWarning,
             stacklevel=3)
        if n_jobs == 1:
            n_jobs = n_cpus
    if backend not in NETWORK_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(NETWORK_BACKENDS)}")
    return backend, n_jobs


def cal_pc_coefs(k: int,
                 X: np.ndarray,
                 n_comp: int,
                 method: str = "sklearn",
                 random_state: int = 42) -> np.ndarray:
    """Regress gene ``k`` on the remaining genes via low-rank SVD.

    Parameters
    ----------
    k
        Index of the response gene.
    X
        Cells-by-genes design matrix (standardized).
    n_comp
        Number of SVD components.
    method
        SVD backend: ``"sklearn"`` (randomized) or ``"scipy"``.
    random_state
        Seed used by the sklearn randomized SVD.

    Returns
    -------
    Column vector of regression coefficients (``(genes - 1, 1)``).
    """
    y = X[:, k]
    Xi = np.delete(X, k, 1)  # cells x (genes - 1)

    if method == "sklearn":
        U, Sigma, VT = randomized_svd(Xi,
                                      n_components=n_comp,
                                      flip_sign=True,  # to yield deterministic outputs
                                      n_iter=20,
                                      random_state=random_state)
    # elif method == "dask":
        # U, Sigma, VT = da.linalg.svd_compressed(da.from_array(Xi), k=n_comp)
        # VT = VT.compute()
    elif method == "scipy":
        U, Sigma, VT = scipy.linalg.svd(Xi, False, lapack_driver="gesvd")
    else:
        raise ValueError("Invalid method")
    coef = VT[:n_comp, :].T  # (genes - 1) x n_comp
    score = Xi @ coef  # cells x n_comp
    score = score / np.expand_dims((score ** 2).sum(axis=0), 0)
    betas = coef @ (score.T @ y)  # (genes - 1),
    return np.expand_dims(betas, 1)


def _pc_net_standardize(X: np.ndarray) -> tuple:
    """Standardize the non-constant genes of a genes x cells matrix.

    Returns the cells x genes matrix of the non-constant genes, each with mean
    0 and standard deviation 1, and a boolean mask of the non-constant genes.
    """
    X_std = np.array(X, dtype=float).T
    n_samples = X_std.shape[0]
    # Exact test: a gene is constant if every value equals its first value
    non_constant = (X_std != X_std[0]).any(axis=0)
    X_std = X_std[:, non_constant]
    X_std = X_std - X_std.mean(axis=0)
    gene_sds = np.sqrt((X_std * X_std).sum(axis=0) / max(1, n_samples - 1))
    return X_std / gene_sds, non_constant


def _pc_net_eigen_basis(X_std: np.ndarray) -> tuple:
    """Eigendecomposition of the Gram matrix ``X_std X_std' = Q diag(d) Q'``.

    Returns ``Z = Q' X_std`` (column k holds gene k in the eigenbasis) and the
    eigenvalues ``d`` in decreasing order. The smaller of the two Gram
    matrices is decomposed; both give the same ``d`` and ``Z``.
    """
    if X_std.shape[0] <= X_std.shape[1]:
        d, Q = np.linalg.eigh(X_std @ X_std.T)
        d, Q = np.maximum(d[::-1], 0), Q[:, ::-1]
        Z = Q.T @ X_std
    else:
        d, V = np.linalg.eigh(X_std.T @ X_std)
        d, V = np.maximum(d[::-1], 0), V[:, ::-1]
        Z = np.sqrt(d)[:, None] * V.T
    return Z, d


def _pc_net_secular_weights(Z: np.ndarray, d: np.ndarray, n_comp: int) -> np.ndarray:
    """Solve the per-gene eigenproblems through the secular equation.

    Column k of the result is ``w = sum_i u_i (u_i' z) / mu_i``, where
    ``(mu_i, u_i)`` are the top ``n_comp`` eigenpairs of ``D - z z'`` and
    ``z`` is column k of ``Z``; the coefficients of gene k are ``Z' w``.
    The i-th root ``mu_i`` is the only root in ``(d[i + 1], d[i])`` of
    ``f(mu) = 1 - sum_j z_j^2 / (d_j - mu)``; it is found as in LAPACK's
    dlaed4, for all genes at once.
    """
    eps = np.finfo(float).eps
    n_eig_input, n_genes = Z.shape
    n_eig = n_eig_input

    # With very few samples, every interval needs a lower end: the remaining
    # eigenvalues of the Gram matrix are zero
    if n_eig < n_comp + 1:
        n_pad = n_comp + 1 - n_eig
        Z = np.vstack([Z, np.zeros((n_pad, n_genes))])
        d = np.concatenate([d, np.zeros(n_pad)])
        n_eig = n_comp + 1

    # A zero z_j means d_j is still an eigenvalue after removing the gene;
    # raising such entries to a negligible size gives the same result through
    # the general formula
    tiny = np.sqrt(max(d[0], np.finfo(float).tiny)) * 1e-60
    Z = Z.copy()
    near_zero = np.abs(Z) < tiny
    Z[near_zero] = np.where(Z[near_zero] < 0, -tiny, tiny)
    Z2 = Z * Z

    W = np.zeros((n_eig, n_genes))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(n_comp):
            d_upper = d[i]
            d_lower = d[i + 1]
            gap = d_upper - d_lower
            # Equal eigenvalues: the eigenvector is orthogonal to z
            if not gap > 0:
                continue

            # Origin at the closer end: f(midpoint) >= 0 means the root is in
            # the upper half of the interval
            midpoint = d_lower + gap / 2
            near_upper = (1 - (Z2 / (d - midpoint)[:, None]).sum(axis=0)) >= 0
            origin = np.where(near_upper, d_upper, d_lower)

            # Everything below is relative to origin (tau = mu - origin)
            pole_lower = np.where(near_upper, -gap, 0.)
            pole_upper = np.where(near_upper, 0., gap)
            bracket_lo = np.where(near_upper, -gap / 2, 0.)
            bracket_hi = np.where(near_upper, 0., gap / 2)
            tau = (bracket_lo + bracket_hi) / 2
            d_minus_origin = d[:, None] - origin[None, :]

            active = np.arange(n_genes)
            for _ in range(300):
                tau_a = tau[active]
                gaps = d_minus_origin[:, active] - tau_a[None, :]
                terms = Z2[:, active] / gaps
                slopes = terms / gaps

                sum_above = terms[:i + 1].sum(axis=0)
                slope_above = slopes[:i + 1].sum(axis=0)
                sum_below = terms[i + 1:].sum(axis=0)
                slope_below = slopes[i + 1:].sum(axis=0)
                f = 1 - sum_above - sum_below

                # f is decreasing: f > 0 means the root is above tau
                lo = bracket_lo[active]
                hi = bracket_hi[active]
                lo[f > 0] = tau_a[f > 0]
                hi[f < 0] = tau_a[f < 0]
                bracket_lo[active] = lo
                bracket_hi[active] = hi

                # One-pole models of both sums, matching value and slope at
                # tau; solving model = 1 gives A step^2 - B step + C = 0
                to_lower = pole_lower[active] - tau_a
                to_upper = pole_upper[active] - tau_a
                A = f + slope_below * to_lower + slope_above * to_upper
                B = A * (to_lower + to_upper) - \
                    slope_below * to_lower * to_lower - slope_above * to_upper * to_upper
                C = f * to_lower * to_upper

                # Numerically stable roots; keep the one between the poles
                sqrt_disc = np.sqrt(np.maximum(B * B - 4 * A * C, 0))
                half = (B + np.where(B >= 0, sqrt_disc, -sqrt_disc)) / 2
                root_1 = half / A
                root_2 = C / half
                step = np.where(~np.isnan(root_1) & (root_1 > to_lower) & (root_1 < to_upper),
                                root_1, root_2)

                # Fall back to bisection if the step leaves the bracket
                tau_new = tau_a + step
                outside = np.isnan(tau_new) | (tau_new <= lo) | (tau_new >= hi)
                tau_new[outside] = (lo[outside] + hi[outside]) / 2

                exact = f == 0
                tau_new[exact] = tau_a[exact]
                converged = exact | \
                    (np.abs(tau_new - tau_a) <= 4 * eps * np.abs(tau_new)) | \
                    ((hi - lo) <= 4 * eps * np.maximum(np.abs(lo), np.abs(hi)))

                tau[active] = tau_new
                active = active[~converged]
                if len(active) == 0:
                    break

            # Add this eigenpair's term v / (mu ||v||^2) with v = z / (d - mu)
            mu = origin + tau
            v = Z / (d_minus_origin - tau[None, :])
            W = W + v * (1 / (mu * (v * v).sum(axis=0)))[None, :]

    return W[:n_eig_input]


def _pc_net_prior_matrix(prior_network: pd.DataFrame, gene_names: Sequence) -> np.ndarray:
    """Dense genes x genes counts of the (regulator, target) prior edges."""
    if prior_network.shape[1] != 2:
        raise ValueError("Prior network needs to be a two column data.frame with regulators and targets")
    position = pd.Index(gene_names).get_indexer
    rows = position(prior_network.iloc[:, 0])
    cols = position(prior_network.iloc[:, 1])
    known = (rows >= 0) & (cols >= 0)
    prior = np.zeros((len(gene_names), len(gene_names)))
    # Edges listed more than once are counted more than once
    np.add.at(prior, (rows[known], cols[known]), 1)
    return prior


def pc_net(X: np.ndarray,
           n_comp: int = 3,
           scale_scores: bool = True,
           symmetric: bool = False,
           q: float = 0.,
           prior_network: Optional[pd.DataFrame] = None,
           gene_names: Optional[Sequence] = None) -> np.ndarray:
    """Principal component regression network, as ``pcNet`` in R.

    Every gene is regressed on the leading ``n_comp`` principal components of
    all the other genes (standardized), and the regression coefficients
    become the edge weights. The coefficients are computed exactly, from a
    single eigendecomposition and the secular equation of each gene.

    Parameters
    ----------
    X
        Genes-by-cells expression matrix.
    n_comp
        Number of principal components per gene regression (>= 2 and lower
        than the number of non-constant genes).
    scale_scores
        If True, divide by the maximum absolute weight.
    symmetric
        If True, return ``(A + A.T) / 2``.
    q
        Quantile in ``[0, 1]``; edges whose absolute weight is below it are
        set to zero. 0 or 1 disable filtering.
    prior_network
        Optional two-column DataFrame of (regulator, target) edges, keeping
        their weight when ``q`` filters edges. Requires ``gene_names``.
    gene_names
        Names of the rows of ``X``, used to match ``prior_network``.

    Returns
    -------
    Dense genes-by-genes matrix; entry ``[i, j]`` is the coefficient of gene
    ``j`` in the regression of gene ``i``. The diagonal is zero, and so are
    the rows and columns of constant genes.
    """
    if n_comp < 2:
        raise ValueError(f"n_comp must be >= 2. Got: {n_comp}")
    n_genes = X.shape[0]

    # Constant genes (e.g. all zeros) are set aside
    X_std, used = _pc_net_standardize(X)
    n_used = int(used.sum())
    if n_comp >= n_used:
        raise ValueError(f"n_comp must be < number of non-constant genes ({n_used}). Got: {n_comp}")

    Z, d = _pc_net_eigen_basis(X_std)
    del X_std
    network = _pc_net_secular_weights(Z, d, n_comp).T @ Z
    del Z
    np.fill_diagonal(network, 0)

    if symmetric:
        network = (network + network.T) / 2

    if scale_scores:
        max_abs_value = np.nanmax(np.abs(network))
        if np.isfinite(max_abs_value) and max_abs_value > 0:
            network = network / max_abs_value

    if 0 < q < 1:
        abs_network = np.abs(network)
        threshold = np.nanquantile(abs_network, q)
        prior_weight = None
        if prior_network is not None:
            if gene_names is None:
                raise ValueError("gene_names are required to use a prior network")
            # Prior edges keep their weight, times the number of times listed
            prior_weight = _pc_net_prior_matrix(prior_network, gene_names)[np.ix_(used, used)] * network
        network[abs_network < threshold] = 0
        if prior_weight is not None:
            restore = prior_weight != 0
            network[restore] = prior_weight[restore]

    if n_used < n_genes:
        full = np.zeros((n_genes, n_genes))
        full[np.ix_(used, used)] = network
        network = full
    return network


def pc_net_calc(data: pd.DataFrame,  # genes x cells
                selected_samples: Union[List[int], np.ndarray],
                n_comp: int = 3,
                scale_scores: bool = True,
                symmetric: bool = False,
                q: float = 0.,
                random_state: Optional[int] = None,
                prior_network: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Compute a single principal-component (PC) network.

    The genes not expressed in any of the selected cells are left out; see
    :func:`pc_net` for the method.

    Parameters
    ----------
    data
        Genes-by-cells expression DataFrame.
    selected_samples
        Cell column indices used to build the network.
    n_comp
        Number of principal components per gene regression (>= 2).
    scale_scores
        If True, divide by the maximum absolute weight.
    symmetric
        If True, return ``(A + A.T) / 2``.
    q
        Quantile cutoff in ``[0, 1]`` below which weights are zeroed.
    random_state
        Deprecated and ignored: the coefficients are computed exactly, so
        there is nothing to seed. The cells are sampled in
        :func:`make_networks`.
    prior_network
        Optional two-column DataFrame of (regulator, target) edges.

    Returns
    -------
    Dense adjacency matrix of the genes expressed in the selected cells.
    """
    if random_state is not None:
        warn("random_state is ignored by pc_net_calc, which computes the network exactly, "
             "and will be removed in a future release. Seed the cell sampling with "
             "make_networks(random_state=...) instead.",
             DeprecationWarning,
             stacklevel=2)
    assert 0 <= q <= 1
    Z = data.iloc[:, selected_samples]
    assert not any(Z.index.duplicated()), "some genes are duplicated"
    Z = Z.loc[Z.sum(axis=1) > 0, :]
    return pc_net(Z.to_numpy(), n_comp=n_comp, scale_scores=scale_scores, symmetric=symmetric,
                  q=q, prior_network=prior_network, gene_names=Z.index)


@timer
def make_networks(data: ExpressionData,
                  n_nets: int = 10,
                  n_samp_cells: Optional[int] = 500,
                  n_comp: int = 3,
                  scale_scores: bool = True,
                  symmetric: bool = False,
                  q: float = 0.95,
                  random_state: Optional[int] = 1,
                  backend: Backend = "serial",
                  n_jobs: int = 1,
                  n_cpus: Optional[int] = None,
                  replace: bool = True,
                  layer: LayerName = None,
                  prior_network: Optional[pd.DataFrame] = None,
                  **kwargs: object
                  ) -> List[coo_matrix]:
    """
    Make PCNets from a data frame by subsampling the cells

    Equivalent to ``set.seed(random_state); makeNetworks(...)`` in the R
    package scTenifoldNet: the cells are drawn with R's random number
    generator, so both give the same networks.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe
    n_nets: int, default = 10
        Number of subsampling times
    n_samp_cells: int, None, default = 500
        Number of sampled cells, if None than select all cells
    n_comp: int, default = 3
        Number of principal components in each gene regression (>= 2)
    scale_scores: bool, default = True
        To scale the final PCNets scores or not
    symmetric: bool, default = False
        To make the final PCNets symmetric or not
    q: float, default = 0.95
        The quantile value used to determine PCNet's threshold
    random_state: int, None, default = 1
        Seed of the cell subsampling, as ``set.seed`` in R. If None, a random
        seed is used.
    backend: str, default = "serial"
        Parallel backend: "serial", "joblib-loky", "joblib-threading", or "ray"
    n_jobs: int, default = 1
        Number of workers for parallel backends. -1 uses the backend default.
    n_cpus: int, optional
        Deprecated alias for n_jobs.
    replace: bool, default = True
        Whether cells are sampled with replacement
    prior_network: pd.DataFrame, optional
        Two-column DataFrame of (regulator, target) edges that keep their
        weight when ``q`` filters edges

    kwargs
        Keyword arguments

    Returns
    -------
    networks: List[coo_matrix]
        A list contains PCNets (in coo sparse matrix format)
    """
    data = anndata_to_dataframe(data, layer=layer)
    backend, n_jobs = _resolve_backend(backend=backend, n_jobs=n_jobs, n_cpus=n_cpus)
    gene_names = data.index.to_numpy()
    n_genes, n_cells = data.shape
    assert not np.array_equal(gene_names, np.array([i for i in range(n_genes)])), 'Gene names are required'
    if n_comp < 2 or n_comp >= n_genes:
        raise ValueError("n_comp should be >= 2 and < total number of genes")
    rng = RRandom(random_state)
    sel_samples = []
    for net in range(n_nets):
        sample = rng.sample(n_cells, n_samp_cells, replace=replace) if n_samp_cells is not None else np.arange(n_cells)
        sel_samples.append(sample)

    network_kws = dict(n_comp=n_comp, scale_scores=scale_scores, symmetric=symmetric, q=q,
                       prior_network=prior_network)
    if backend == "serial":
        results = [pc_net_calc(data, selected_samples=sample, **network_kws) for sample in sel_samples]
    elif backend == "joblib-loky":
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(pc_net_calc)(data, selected_samples=sample, **network_kws)
            for sample in sel_samples
        )
    elif backend == "joblib-threading":
        from joblib import Parallel, delayed
        from threadpoolctl import threadpool_limits
        # Concurrent calls into a multithreaded BLAS (OpenBLAS in particular)
        # can crash or hang, so each thread gets a single-threaded BLAS
        with threadpool_limits(limits=1, user_api="blas"):
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(pc_net_calc)(data, selected_samples=sample, **network_kws)
                for sample in sel_samples
            )
    else:
        try:
            from importlib import import_module
            ray = import_module("ray")
        except ImportError as exc:
            raise ImportError("Install scTenifoldpy[parallel-ray] to use backend='ray'.") from exc

        @ray.remote
        def _pc_net_ray(data, selected_samples):
            return pc_net_calc(data=data, selected_samples=selected_samples, **network_kws)

        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=None if n_jobs == -1 else n_jobs)
        z_data = ray.put(data)
        tasks = [_pc_net_ray.remote(z_data, sample) for sample in sel_samples]
        results = ray.get(tasks)
        del z_data
        if ray.is_initialized():
            ray.shutdown()

    # Genes not expressed in the sampled cells get no edges
    values = data.to_numpy()
    networks = []
    for sample, pc_net_ in zip(sel_samples, results):
        expressed = np.flatnonzero(values[:, sample].sum(axis=1) > 0)
        full = np.zeros((n_genes, n_genes))
        full[np.ix_(expressed, expressed)] = pc_net_
        networks.append(coo_matrix(full))
    del results
    return networks


@timer
def cal_pcNet(data: ExpressionData,
              n_comp: int = 3,
              scale_scores: bool = True,
              symmetric: bool = False,
              q: float = 0.95,
              random_state: int = 1,
              **kwargs: object
              ) -> coo_matrix:
    """
    Calculate one pcNet without sampling. An API for getting one PCNet instead of many.

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe
    n_comp: int, default = 3
        Number of PCNets composition
    scale_scores: bool, default = True
        To scale the final PCNets scores or not
    symmetric: bool, default = False
        To make the final PCNets symmetric or not
    q: float, default = 0.95
        The quantile value used to determine PCNet's threshold
    random_state: int, default = 1
        Unused, as all cells are used; kept so that existing calls keep working
    kwargs
        Keyword arguments

    Returns
    -------
    pcNet: coo_matrix
    Result network

    See Also
    --------
    make_networks

    """
    return make_networks(data,
                         n_nets=1,
                         n_samp_cells=None,
                         n_comp=n_comp,
                         scale_scores=scale_scores,
                         symmetric=symmetric, q=q,
                         random_state=random_state, **kwargs)[0]


@timer
def manifold_alignment(X: pd.DataFrame,
                       Y: pd.DataFrame,
                       d: int = 30,
                       tol: float = 1e-8,
                       **kwargs: object
                       ) -> pd.DataFrame:
    """
    Performing manifold alignment on two dataframes

    Parameters
    ----------
    X: pd.DataFrame
        A gene regulatory network X, expected shape = (n_genes, n_genes)
    Y: pd.DataFrame
        A gene regulatory network Y, expected shape = (n_genes, n_genes)
    d: int, default = 30
        The dimension of the low-dimensional feature space
    tol: float, default = 1e-8
        Eigenvectors with eigenvalues not above this tolerance are dropped
    Returns
    -------
    ma_df: pd.DataFrame
        A dataframe contains manifold alignment result, expected shape = (n_genes * 2, d)
    """
    y_genes = set(Y.index)
    shared_genes = [gene for gene in X.index if gene in y_genes]
    if len(shared_genes) == 0:
        raise ValueError("X and Y do not share any genes")
    X = X.loc[shared_genes, shared_genes]
    Y = Y.loc[shared_genes, shared_genes]
    L = np.eye(len(shared_genes))
    w_X, w_Y = X.values + 1, Y.values + 1
    w_XY = L * (0.9 * (np.sum(w_X) + np.sum(w_Y)) / (2 * len(shared_genes)))
    W = -np.concatenate((np.concatenate((w_X, w_XY), axis=1),
                         np.concatenate((w_XY.T, w_Y), axis=1)), axis=0)
    np.fill_diagonal(W, 0)
    np.fill_diagonal(W, -W.sum(axis=0))
    k = d * 2
    if k >= W.shape[0] - 1:
        raise ValueError(f"d={d} is too large for {len(shared_genes)} shared genes; choose d < {(W.shape[0] - 1) / 2}.")
    eg_vals, eg_vecs = scipy.sparse.linalg.eigs(W, k=k, which="SR", tol=1e-14)
    eg_vals = eg_vals.real
    eg_vecs = eg_vecs.real
    order = np.argsort(eg_vals, kind="stable")
    eg_vals, eg_vecs = eg_vals[order], eg_vecs[:, order]
    eg_vecs = eg_vecs[:, eg_vals > tol]
    return pd.DataFrame(eg_vecs[:, :d],
                        index=["X_{g}".format(g=g) for g in shared_genes]+["Y_{g}".format(g=g) for g in shared_genes],
                        columns=["NLMA_{i}".format(i=i+1) for i in range(min(d, eg_vecs.shape[1]))])


def _boxcox_lambda(y: np.ndarray) -> float:
    """Box-Cox power maximizing the profile log-likelihood of ``y ~ 1``.

    Same as ``MASS::boxcox(y ~ 1, lambda = seq(-2, 2, length.out = 1000))``
    followed by taking the lambda of the largest log-likelihood.
    """
    if np.any(y <= 0):
        raise ValueError("response variable must be positive")
    n = len(y)
    lambdas = np.linspace(-2, 2, 1000)
    lambdas = lambdas[lambdas != 0]
    # Scaled by the geometric mean, for accuracy in y^lambda - 1
    y = y / np.exp(np.mean(np.log(y)))
    log_y = np.log(y)
    log_lik = np.empty(len(lambdas))
    for i, la in enumerate(lambdas):
        if abs(la) > 1 / 50:
            yt = (y ** la - 1) / la
        else:
            yt = log_y * (1 + (la * log_y) / 2 * (1 + (la * log_y) / 3 * (1 + (la * log_y) / 4)))
        log_lik[i] = -n / 2 * np.log(np.sum((yt - yt.mean()) ** 2))
    return lambdas[np.argmax(log_lik)]


@timer
def d_regulation(data: pd.DataFrame,
                 sorted_by: Union[str, list] = "p-value",
                 ascending: Union[bool, list] = True,
                 **kwargs: object) -> pd.DataFrame:
    """
    Evaluates the difference in regulation

    Same as ``dRegulation`` in the R packages: the distance of each gene
    between conditions is raised to the Box-Cox power that best normalizes
    the distances, and standardized into Z-scores. P-values follow the
    chi-square distribution of the squared distance relative to its mean.
    Genes whose distance is at the level of floating-point noise (at most
    ``sqrt(eps)`` times the largest absolute coordinate) did not move between
    conditions and get a p-value of 1.

    Parameters
    ----------
    data: pd.DataFrame
        A dataframe contains manifold alignment results, expected shape = (n_genes * 2, d)
    sorted_by: str or list of str, default = "p-value"
        Name or list of names to sort by
    ascending: bool or list of bool, default = True
        Sorted ascending (otherwise descending)
    **kwargs
        Keyword arguments for statistic analyses, and n_ko_genes (if any)
            boxcox_kws - {"lmbda": value} fixes the Box-Cox power
            chi2_kws - kwargs for chi-square test
            n_ko_genes - int, number of largest distances left out of the
                expectation (default 0, as in R)

    Examples
    ---------
    d_reg_df = d_regulation(ma_df)

    d_reg_df = d_regulation(ma_df, boxcox_kws={"lmbda": 0.5}, chi2_kws={"df": 1})

    Returns
    -------
    d_reg_df: pd.DataFrame
        A dataFrame contains difference in regulation result sorted by p-value
        columns: ["Gene", "Distance", "boxcox-transformed distance", "Z", "FC", "p-value", "adjusted p-value"]

    """
    all_gene_names = data.index.to_list()
    gene_names = [g[2:] for g in all_gene_names if "X_" == g[:2]]
    assert len(gene_names) * 2 == len(all_gene_names), 'Number of identified and expected genes are not the same'
    assert all(["Y_" + g == y for g, y in zip(gene_names, all_gene_names[len(gene_names):])]), \
        'Genes are not ordered as expected. X_ genes should be followed by Y_ genes in the same order'
    values = data.to_numpy(dtype=float)
    n_genes = len(gene_names)
    d_metrics = np.sqrt(((values[:n_genes] - values[n_genes:]) ** 2).sum(axis=1))
    boxcox_kws = kwargs.get("boxcox_kws") if "boxcox_kws" in kwargs else {}
    chi2_kws = dict(kwargs.get("chi2_kws")) if "chi2_kws" in kwargs else {}
    if "df" not in chi2_kws:
        chi2_kws["df"] = 1

    # Box-Cox power; the distances are left as they are if it cannot be found.
    # With a negative power, zero distances give 1 / inf = 0, as in R.
    try:
        bc_lambda = boxcox_kws["lmbda"] if "lmbda" in boxcox_kws else _boxcox_lambda(d_metrics)
        with np.errstate(divide="ignore"):
            t_d_metrics = 1 / (d_metrics ** bc_lambda) if bc_lambda < 0 else d_metrics ** bc_lambda
    except ValueError:
        t_d_metrics = d_metrics

    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = (t_d_metrics - t_d_metrics.mean()) / t_d_metrics.std(ddof=1)
        n_ko_genes = kwargs.get("n_ko_genes") if "n_ko_genes" in kwargs else 0
        expected_val = np.mean(np.power(d_metrics[np.argsort(d_metrics)[::-1][n_ko_genes:]], 2))
        FC = np.power(d_metrics, 2) / expected_val
    p_values = stats.chi2.sf(FC, **chi2_kws)

    # Distances at the level of floating-point noise mean the gene did not
    # move between conditions; ranking them against each other would flag noise
    noise_level = np.sqrt(np.finfo(float).eps) * np.max(np.abs(values))
    is_noise = d_metrics <= noise_level
    p_values[is_noise] = 1
    if is_noise.all():
        warn("No gene differs between the two conditions beyond numerical noise; all p-values were set to 1")
    p_adj = cal_fdr(p_values)
    df = pd.DataFrame({
        "Gene": gene_names,
        "Distance": d_metrics,
        "boxcox-transformed distance": t_d_metrics,
        "Z": z_scores,
        "FC": FC,
        "p-value": p_values,
        "adjusted p-value": p_adj
    })
    return df.sort_values(sorted_by, ascending=ascending, kind="mergesort")


def strict_direction(data: np.ndarray, lambd: float = 1) -> np.ndarray:
    """Enforce edge directionality by zeroing the weaker of each ``(i, j)`` / ``(j, i)`` pair.

    Parameters
    ----------
    data
        Square adjacency matrix.
    lambd
        Interpolation weight between the original and strict matrix
        (0 = original, 1 = strict).

    Returns
    -------
    Adjacency with directionality applied.
    """
    if lambd == 0:
        return data
    s_data = data.copy()
    s_data[abs(s_data) < abs(s_data.T)] = 0
    return (1-lambd) * data + lambd * s_data
