"""Heat-kernel tools for virtual knockouts, same as heatKernel, hkManifoldAlignment and
knockoutDirection in the R package scTenifoldKnk (>= 2.0.0)."""
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = ["heat_kernel", "hk_manifold_alignment", "knockout_direction"]

KOGenes = Union[str, Iterable[Union[str, Sequence[str]]]]


def heat_kernel(data: Union[np.ndarray, pd.DataFrame],
                t: float = 10,
                symmetric: bool = True) -> Union[np.ndarray, pd.DataFrame]:
    """Spectral heat kernel of a gene-gene matrix.

    Computes ``H = sum_k exp(t * (lambda_k / lambda_max - 1)) v_k v_k^T`` from the
    eigendecomposition of a symmetric gene-gene matrix (a gene regulatory network
    or a gene-gene correlation matrix). Row ``i`` of ``H`` describes how a change
    in gene ``i`` diffuses over the whole network. Same as ``heatKernel`` in R.

    Parameters
    ----------
    data
        Square gene-gene matrix.
    t
        Non-negative diffusion time; ``t = 0`` returns the identity.
    symmetric
        If True, ``data`` is replaced by its symmetric part ``(A + A.T) / 2``
        first; required for directed networks such as the scTenifoldKnk WT
        network.

    Returns
    -------
    The heat kernel, with the labels of ``data`` when it is a DataFrame.
    """
    if not np.isscalar(t) or t < 0:
        raise ValueError("'t' must be a single non-negative number")
    A = np.asarray(data, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("'data' must be a square matrix")
    if symmetric:
        A = (A + A.T) / 2
    elif not np.allclose(A, A.T):
        raise ValueError("'data' is not symmetric; use symmetric=True")
    lam, V = np.linalg.eigh(A)
    if lam.max() <= 0:
        raise ValueError("The largest eigenvalue of 'data' must be positive")
    H = (V * np.exp(t * (lam / lam.max() - 1))) @ V.T
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(H, index=data.index, columns=data.columns)
    return H


def _ko_list(ko_genes: KOGenes) -> List[List[str]]:
    # A string is one gene; each element of an iterable is a gene or a group knocked out together
    if isinstance(ko_genes, str):
        return [[ko_genes]]
    return [[g] if isinstance(g, str) else list(g) for g in ko_genes]


def _log_cpm(cpm: pd.DataFrame, genes: Sequence[str], lib_size: np.ndarray):
    # log1p(CPM) of the genes, their mean and standard deviation (as .logCPM in R)
    missing = [g for g in genes if g not in cpm.index]
    if missing:
        raise ValueError(f"The following genes are not present in the data: {missing}")
    L = np.log1p(cpm.loc[list(genes)].to_numpy(dtype=float))
    mu = L.mean(axis=1)
    sd = np.sqrt(((L - mu[:, None]) ** 2).mean(axis=1))
    sd[sd == 0] = np.nan
    return {"L": L, "mu": mu, "sd": sd, "lib_size": np.asarray(lib_size, dtype=float)}


def _counts_stats(counts: pd.DataFrame, genes: Sequence[str]):
    counts = counts.astype(float)
    lib_size = counts.sum(axis=0).to_numpy()
    return _log_cpm(counts / lib_size * 1e6, genes, lib_size)


def _regress_lib_size(stats) -> np.ndarray:
    # Residuals of each gene's log1p(CPM) after a least-squares fit on log library
    # size (with intercept), removing the sequencing-depth axis from the correlations
    ll = np.log(stats["lib_size"])
    ll = ll - ll.mean()
    Lc = stats["L"] - stats["mu"][:, None]
    ss = (ll ** 2).sum()
    if ss == 0:
        return Lc
    return Lc - np.outer(Lc @ ll / ss, ll)


def _kernel_effect(H: np.ndarray, stats, genes: pd.Index, kos: List[List[str]]) -> np.ndarray:
    # Signed change of every gene when the genes of each knockout are removed:
    # sum over the knocked-out genes x of -mean(x) / sd(x) * H[x, g] * sd(g)
    out = np.zeros((len(kos), len(genes)))
    sd = np.nan_to_num(stats["sd"])
    for k, ko in enumerate(kos):
        idx = genes.get_indexer(ko)
        w = np.nan_to_num(-stats["mu"][idx] / stats["sd"][idx])
        out[k] = (w @ H[idx]) * sd
    return np.nan_to_num(out)


def _check_genes(kos: List[List[str]], genes: pd.Index, where: str) -> None:
    missing = sorted({g for ko in kos for g in ko} - set(genes))
    if missing:
        raise ValueError(f"The following genes are not present in {where}: {missing}")


def _hk_distances(wt: pd.DataFrame, stats, kos: List[List[str]], t: float = 10,
                  H: Optional[np.ndarray] = None) -> pd.DataFrame:
    genes = pd.Index(wt.index)
    if H is None:
        W = wt.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(W, 0)
        H = heat_kernel(W, t=t)
    D = np.abs(_kernel_effect(np.asarray(H), stats, genes, kos))
    # A knocked-out gene loses its whole expression: its change is its mean log1p(CPM)
    for k, ko in enumerate(kos):
        idx = genes.get_indexer(ko)
        D[k, idx] = stats["mu"][idx]
    return pd.DataFrame(D, index=["+".join(ko) for ko in kos], columns=genes)


def _direction_scores(stats, genes: pd.Index, kos: List[List[str]], t: float = 5,
                      regress_lib_size: bool = False) -> pd.DataFrame:
    X = _regress_lib_size(stats) if regress_lib_size else stats["L"]
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.corrcoef(X)
    R = np.nan_to_num(R)
    H = heat_kernel(R, t=t)
    return pd.DataFrame(_kernel_effect(H, stats, genes, kos),
                        index=["+".join(ko) for ko in kos], columns=genes)


def hk_manifold_alignment(wt: pd.DataFrame,
                          counts: pd.DataFrame,
                          ko_genes: Optional[KOGenes] = None,
                          t: float = 10,
                          H: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Heat-kernel (hk) manifold alignment for in-silico knockouts.

    Kernel counterpart of the manifold alignment: the heat kernel of the WT
    network is computed once (:func:`heat_kernel`) and each knockout is read
    from its rows. Removing gene ``x`` lowers it by its mean expression and the
    change diffuses over the network, ``Delta_g = -mean(x) / sd(x) * H[x, g] * sd(g)``
    on log1p(CPM) expression (summed over the genes of a multi-gene knockout).
    ``|Delta_g|`` plays the role of the manifold alignment distance. Same as
    ``hkManifoldAlignment`` in R.

    Parameters
    ----------
    wt
        WT gene regulatory network (genes x genes, regulators as rows), e.g.
        ``scTenifoldKnk.tensor_dict["WT"]``.
    counts
        Raw counts (genes x cells) of the WT cells, with the genes of ``wt``
        among its rows (typically the quality-controlled counts).
    ko_genes
        Gene, or iterable of genes (or of lists of genes knocked out together).
        ``None`` knocks out every gene of the network separately.
    t
        Diffusion time of the heat kernel.
    H
        Optional pre-computed heat kernel of ``wt`` to reuse across calls.

    Returns
    -------
    Perturbation distances, one row per knockout (genes joined with ``"+"``)
    and one column per gene of the network. For the knocked-out genes the
    value is their mean log1p(CPM) expression, the amount they lose.
    """
    genes = pd.Index(wt.index)
    kos = [[g] for g in genes] if ko_genes is None else _ko_list(ko_genes)
    _check_genes(kos, genes, "the WT network")
    return _hk_distances(wt, _counts_stats(counts, genes), kos, t=t, H=H)


def knockout_direction(counts: pd.DataFrame,
                       ko_genes: KOGenes,
                       genes: Optional[Sequence[str]] = None,
                       t: float = 5,
                       regress_lib_size: bool = False) -> pd.DataFrame:
    """Direction of the response to an in-silico knockout.

    Predicts whether each gene goes up or down after knocking out ``ko_genes``
    from the WT expression only: the heat kernel of the gene-gene Pearson
    correlation of log1p(CPM) expression is diffused from the knocked-out
    gene(s), ``s_g = -sum_x mean(x) / sd(x) * H[x, g] * sd(g)``, and the sign of
    ``s_g`` is the predicted direction. Same as ``knockoutDirection`` in R.

    The direction mostly reflects the response shared by most knockdowns along
    the dominant WT expression program rather than regulation specific to the
    knocked-out gene, and its accuracy varies between cell types and data sets.
    In single-cell data, log1p(CPM) expression often keeps a dependence on
    sequencing depth that makes nearly all genes correlate positively, so the
    kernel predicts almost every gene to go down; ``regress_lib_size=True``
    regresses log library size out of each gene before computing the
    correlations, which removes that axis. It is off by default: it corrected
    tissue data where nearly all genes were predicted down, but in cell lines
    it left the direction unchanged or made it slightly worse, because the
    expression that follows library size can be biological.

    Parameters
    ----------
    counts
        Raw counts (genes x cells) of the WT cells.
    ko_genes
        Gene, or iterable of genes (or of lists of genes knocked out together).
    genes
        Genes to score (e.g. the genes of the WT network). Default: all genes
        of ``counts``.
    t
        Diffusion time of the heat kernel.
    regress_lib_size
        Regress the log library size of each cell (column sums of ``counts``)
        out of each gene's log1p(CPM) before computing the correlations.
        Default False.

    Returns
    -------
    Direction scores (positive: predicted up, negative: predicted down), one
    row per knockout and one column per gene.
    """
    genes = pd.Index(counts.index if genes is None else genes)
    kos = _ko_list(ko_genes)
    _check_genes(kos, genes, "'genes'")
    return _direction_scores(_counts_stats(counts, genes), genes, kos, t=t, regress_lib_size=regress_lib_size)
