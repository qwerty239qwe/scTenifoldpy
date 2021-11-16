from functools import partial
from warnings import warn
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix
import scipy.sparse.linalg
from sklearn.utils.extmath import randomized_svd
import ray

from scTenifold.core._utils import cal_fdr, timer


__all__ = ["make_networks", "cal_pcNet", "cal_pc_coefs", "manifold_alignment", "d_regulation", "strict_direction"]


def cal_pc_coefs(k, X, n_comp, method="sklearn", random_state=42):

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
    score = score / np.expand_dims(np.power(np.sqrt(np.sum(np.power(score, 2), axis=0)), 2), 0)
    betas = coef.dot(np.sum(np.expand_dims(y, 1) * score, axis=0))  # (genes - 1),
    return np.expand_dims(betas, 1)


def _check_pcNet_inp(data, selected_samples):
    Z = data.iloc[:, selected_samples]
    sel_genes = (Z.sum(axis=1) > 0)
    assert not any(sel_genes.index.duplicated()), "some genes are duplicated"
    Z = Z.loc[sel_genes, :]
    assert all(Z.sum(axis=1) > 0), "All genes must be expressed in at least one cell"
    return Z.values


def pc_net_calc(data: pd.DataFrame,  # genes x cells
                selected_samples,
                n_comp: int = 3,
                scale_scores: bool = True,
                symmetric: bool = False,
                q: float = 0.,
                random_state: int = 42):
    assert 2 < n_comp <= data.shape[0]
    assert 0 <= q <= 1
    X = _check_pcNet_inp(data, selected_samples)
    Xt = X.T  # cells x genes
    Xt = (Xt - Xt.mean(axis=0)) / Xt.std(axis=0)
    A = 1 - np.eye(Xt.shape[1])  # genes x genes
    p_ = partial(cal_pc_coefs, X=Xt, n_comp=n_comp, random_state=random_state)
    bs = [p_(i) for i in range(Xt.shape[1])]
    B = np.concatenate(bs, axis=1).T  # beta matrix ((genes - 1), genes)

    A[A > 0] = np.ravel(B)
    if symmetric:
        A = (A + A.T) / 2
    abs_A = abs(A)
    if scale_scores:
        A = A / np.max(abs_A)
    A[abs_A < np.quantile(abs_A, q)] = 0
    np.fill_diagonal(A, 0)
    return A


@ray.remote
def pc_net_parallelized(data: pd.DataFrame,  # genes x cells
                        selected_samples,
                        n_comp: int = 3,
                        scale_scores: bool = True,
                        symmetric: bool = False,
                        q: float = 0.,
                        random_state: int = 42):
    return pc_net_calc(data=data,
                       selected_samples=selected_samples,
                       n_comp=n_comp,
                       scale_scores=scale_scores,
                       symmetric=symmetric,
                       q=q,
                       random_state=random_state)


def pc_net_single(data: pd.DataFrame,  # genes x cells
                  selected_samples,
                  n_comp: int = 3,
                  scale_scores: bool = True,
                  symmetric: bool = False,
                  q: float = 0.,
                  random_state: int = 42):
    return pc_net_calc(data=data,
                       selected_samples=selected_samples,
                       n_comp=n_comp,
                       scale_scores=scale_scores,
                       symmetric=symmetric,
                       q=q,
                       random_state=random_state)


@timer
def make_networks(data: pd.DataFrame,
                  n_nets: int = 10,
                  n_samp_cells: Optional[int] = 500,
                  n_comp: int = 3,
                  scale_scores: bool = True,
                  symmetric: bool = False,
                  q: float = 0.95,
                  random_state: int = 42,
                  n_cpus: int = -1,
                  **kwargs
                  ) -> List[coo_matrix]:
    """
    Make PCNets from a data frame by subsampling the cells

    Parameters
    ----------
    data: pd.DataFrame
        Input dataframe
    n_nets: int, default = 10
        Number of subsampling times
    n_samp_cells: int, None, default = 500
        Number of sampled cells, if None than select all cells
    n_comp: int, default = 3
        Number of PCNets composition
    scale_scores: bool, default = True
        To scale the final PCNets scores or not
    symmetric: bool, default = False
        To make the final PCNets symmetric or not
    q: float, default = 0.95
        The quantile value used to determine PCNet's threshold
    random_state: int, default = 42
        Random seed of constructing PCNets
    n_cpus: int, default = -1

    kwargs
        Keyword arguments

    Returns
    -------
    networks: List[coo_matrix]
        A list contains PCNets (in coo sparse matrix format)
    """
    gene_names = data.index.to_numpy()
    n_genes, n_cells = data.shape
    assert not np.array_equal(gene_names, np.array([i for i in range(n_genes)])), 'Gene names are required'
    rng = np.random.default_rng(random_state)
    networks = []
    sel_samples = []
    tasks = []

    if n_cpus != 1:
        if n_cpus == -1:
            n_cpus = None
        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=n_cpus)
        # dask.config.set(scheduler=ray_dask_get)
        Z_data = ray.put(data)
        for net in range(n_nets):
            sample = rng.choice(n_cells, n_samp_cells, replace=True) if n_samp_cells is not None else np.arange(n_cells)
            sel_samples.append(sample)
            tasks.append(pc_net_parallelized.remote(Z_data,
                                                    selected_samples=sample,
                                                    n_comp=n_comp,
                                                    scale_scores=scale_scores,
                                                    symmetric=symmetric,
                                                    q=q,
                                                    random_state=random_state))
        results = ray.get(tasks)
        del Z_data
    else:
        results = []
        for net in range(n_nets):
            sample = rng.choice(n_cells, n_samp_cells, replace=True) if n_samp_cells is not None else np.arange(n_cells)
            sel_samples.append(sample)
            results.append(pc_net_single(data,
                                         selected_samples=sample,
                                         n_comp=n_comp,
                                         scale_scores=scale_scores,
                                         symmetric=symmetric,
                                         q=q,
                                         random_state=random_state))
    for i, pc_net in enumerate(results):
        Z = data.iloc[:, sel_samples[i]]
        sel_genes = (Z.sum(axis=1) > 0)
        Z = Z.loc[sel_genes, :]
        temp_df = pd.DataFrame(columns=gene_names, index=gene_names)
        temp_df.loc[sel_genes, sel_genes] = pd.DataFrame(pd.DataFrame(pc_net,
                                                                      index=Z.index,
                                                                      columns=Z.index),
                                                         index=sel_genes.index,
                                                         columns=sel_genes.index)
        networks.append(coo_matrix(temp_df.fillna(0.0).values))
    del results
    if ray.is_initialized():
        ray.shutdown()
    return networks


@timer
def cal_pcNet(data: pd.DataFrame,
              n_comp: int = 3,
              scale_scores: bool = True,
              symmetric: bool = False,
              q: float = 0.95,
              random_state: int = 42,
              **kwargs
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
    random_state: int, default = 42
        Random seed of constructing PCNets
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
                       **kwargs
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
        The tolerance of eigen values
    Returns
    -------
    ma_df: pd.DataFrame
        A dataframe contains manifold alignment result, expected shape = (n_genes * 2, d)
    """
    shared_genes = list(set(X.index) & set(Y.index))
    X = X.loc[shared_genes, shared_genes]
    Y = Y.loc[shared_genes, shared_genes]
    L = np.eye(len(shared_genes))
    w_X, w_Y = X.values + 1, Y.values + 1
    w_XY = L * (0.9 * (np.sum(w_X) + np.sum(w_Y)) / (2 * len(shared_genes)))
    W = -np.concatenate((np.concatenate((w_X, w_XY), axis=1),
                         np.concatenate((w_XY.T, w_Y), axis=1)), axis=0)
    np.fill_diagonal(W, 0)
    np.fill_diagonal(W, -W.sum(axis=0))
    eg_vals, eg_vecs = scipy.sparse.linalg.eigs(W, k=d * 2, which="SR")
    eg_vecs = eg_vecs[:, eg_vals >= tol]
    eg_vecs = eg_vecs[:, np.argsort(eg_vals[eg_vals >= tol], )[::-1]]
    return pd.DataFrame(eg_vecs[:, :d],
                        index=["X_{g}".format(g=g) for g in shared_genes]+["Y_{g}".format(g=g) for g in shared_genes],
                        columns=["NLMA_{i}".format(i=i) for i in range(1, 1+d)])


@timer
def d_regulation(data,
                 sorted_by: Union[str, list] = "p-value",
                 ascending: Union[bool, list] = True,
                 **kwargs):
    """
    Evaluates the difference in regulation

    Parameters
    ----------
    data: pd.DataFrame
        A dataframe contains manifold alignment results, expected shape = (n_genes * 2, d)
    sorted_by: str or list of str, default = "p-value"
        Name or list of names to sort by
    ascending: bool or list of bool, default = True
        Sorted ascending (otherwise descending)
    **kwargs
        Keyword arguments for statistic analyses,
            boxcox_kws - kwargs for boxcox test
            chi2_kws - kwargs for chi-square test

    Examples
    ---------
    d_reg_df = d_regulation(ma_df)

    d_reg_df = d_regulation(ma_df, boxcox_kws={"lmbda": 0}, chi2_kws={"df": 1})

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
    d_metrics = np.array([np.linalg.norm((data.iloc[x, :] - data.iloc[y, :]).values)
                          for x, y in zip(range(len(gene_names)),
                                          range(len(gene_names), len(all_gene_names)))])
    boxcox_kws = kwargs.get("boxcox_kws") if "boxcox_kws" in kwargs else {}
    chi2_kws = kwargs.get("chi2_kws") if "chi2_kws" in kwargs else {}
    if "df" not in chi2_kws:
        chi2_kws["df"] = 1
    try:
        t_d_metrics = np.array(stats.boxcox(d_metrics[d_metrics > 0], **boxcox_kws)[0])
    except:
        warn("cannot find the box-cox transformed values")
        t_d_metrics = d_metrics

    z_scores = (t_d_metrics - t_d_metrics.mean()) / t_d_metrics.std()
    expected_val = np.mean(np.power(d_metrics, 2))
    FC = np.power(d_metrics, 2) / expected_val
    p_values = 1 - stats.chi2.cdf(FC, **chi2_kws)
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
    return df.sort_values(sorted_by, ascending=ascending)


def strict_direction(data, lambd=1):
    if lambd == 0:
        return data
    s_data = data.copy()
    s_data[abs(s_data) < abs(s_data.T)] = 0
    return (1-lambd) * data + lambd * s_data
