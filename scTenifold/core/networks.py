import numpy as np
import pandas as pd
from scipy import stats
import scipy.sparse.linalg
from functools import partial
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd
from warnings import warn
from scTenifold.core.utils import cal_fdr

__all__ = ("make_networks", "manifold_alignment", "d_regulation", "strict_direction")


def cal_pc_coefs(k, X, n_comp, random_state=42):
    y = X[:, k]
    Xi = np.delete(X, k, 1)  # cells x (genes - 1)
    U, Sigma, V = randomized_svd(Xi,
                                 n_components=n_comp,
                                 n_iter=5,
                                 random_state=random_state)
    coef = V.T  # (genes - 1) x n_comp
    score = Xi.dot(coef)  # cells x n_comp
    score = score / np.expand_dims(np.power(np.sqrt(np.sum(np.power(score, 2), axis=0)), 2), 0)
    betas = coef.dot(np.sum(np.expand_dims(y, 1) * score, axis=0))  # (genes - 1),
    return np.expand_dims(betas, 1)


def pcNet(X: np.ndarray,  # genes x cells
          n_comp: int = 3,
          scale_scores: bool = True,
          symmetric: bool = False,
          q: float = 0.,
          random_state: int = 42):

    assert 2 < n_comp <= X.shape[0]
    assert 0 <= q <= 1
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


def make_networks(data: pd.DataFrame,
                  n_nets: int = 10,
                  n_samp_cells: int = 500,
                  n_comp: int = 3,
                  scale_scores: bool = True,
                  symmetric: bool = False,
                  q: float = 0.95,
                  random_state: int = 42
                  ):
    gene_names = data.index.to_numpy()
    n_genes, n_cells = data.shape
    assert not np.array_equal(gene_names, np.array([i for i in range(n_genes)])), 'Gene names are required'
    rng = np.random.default_rng(random_state)
    networks = np.empty((n_genes, n_genes, n_nets), dtype=np.float32)
    for net in tqdm(range(n_nets)):
        sample = rng.choice(n_cells, n_samp_cells, replace=False)
        Z = data.iloc[:, sample]
        sel_genes = (Z.sum(axis=1) > 0)
        Z = Z.loc[sel_genes, :]
        temp_df = pd.DataFrame(columns=gene_names, index=gene_names)
        assert all(Z.sum(axis=1) > 0), "All genes must be expressed in at least one cell"
        temp_df.loc[sel_genes, sel_genes] = pd.DataFrame(pd.DataFrame(pcNet(Z.values,
                                                                            n_comp=n_comp,
                                                                            scale_scores=scale_scores,
                                                                            symmetric=symmetric,
                                                                            q=q,
                                                                            random_state=random_state),
                                                                      index=Z.index,
                                                                      columns=Z.index),
                                                         index=sel_genes.index,
                                                         columns=sel_genes.index)
        networks[:, :, net] = temp_df.fillna(0.0).values
    return networks


def manifold_alignment(X: pd.DataFrame,
                       Y: pd.DataFrame,
                       d: int = 30,
                       tol: float = 1e-8):
    shared_genes = list(set(X.index) & set(Y.index))
    X = X.loc[shared_genes, shared_genes]
    Y = Y.loc[shared_genes, shared_genes]
    L = np.eye(len(shared_genes))
    w_X, w_Y = X.values + 1, Y.values + 1
    w_XY = L * (0.9 * (np.sum(w_X) + np.sum(w_Y)) / (2 * len(shared_genes)))
    W = -np.concatenate((np.concatenate((w_X, w_XY), axis=1),
                         np.concatenate((w_XY.T, w_Y), axis=1)), axis=0)
    np.fill_diagonal(W, 0)
    np.fill_diagonal(W, -W.sum(axis=1))
    eg_vals, eg_vecs = scipy.sparse.linalg.eigs(W, k=d * 2, which="SR")
    print(eg_vecs.shape, eg_vals)
    eg_vecs = eg_vecs[:, eg_vals >= tol]
    eg_vecs = eg_vecs[:, np.argsort(eg_vals[eg_vals >= tol], )[::-1]]
    return pd.DataFrame(eg_vecs[:, :d],
                        index=["X_{g}".format(g=g) for g in shared_genes]+["Y_{g}".format(g=g) for g in shared_genes],
                        columns=["NLMA_{i}".format(i=i) for i in range(1, 1+d)])


def d_regulation(data):
    all_gene_names = data.index.to_list()
    gene_names = [g[2:] for g in all_gene_names if "X_" == g[:2]]
    assert len(gene_names) * 2 == len(all_gene_names), 'Number of identified and expected genes are not the same'
    assert all(["Y_" + g == y for g, y in zip(gene_names, all_gene_names[len(gene_names):])]), 'Genes are not ordered as expected. X_ genes should be followed by Y_ genes in the same order'
    d_metrics = np.array([np.linalg.norm((data.iloc[x, :] - data.iloc[y, :]).values)
                          for x, y in zip(range(len(gene_names)),
                                          range(len(gene_names), len(all_gene_names)))])
    try:
        t_d_metrics = np.array(stats.boxcox(d_metrics)[0])
    except:
        warn("cannot find the box-cox transformed values")
        t_d_metrics = d_metrics

    z_scores = (t_d_metrics - t_d_metrics.mean()) / t_d_metrics.std()
    expected_val = np.mean(np.power(d_metrics, 2))
    FC = np.power(d_metrics, 2) / expected_val
    p_values = 1 - stats.chi2.cdf(FC, df=1)
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
    return df.sort_values("p-value", ascending=True)


def strict_direction(data, lambd=1):
    if lambd == 0:
        return data
    s_data = data.copy()
    s_data[abs(s_data) < abs(s_data.T)] = 0
    return (1-lambd) * data + lambd * s_data
