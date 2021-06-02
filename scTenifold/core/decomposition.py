import numpy as np
import pandas as pd
import scipy
from scTenifold.core.utils import timer


def unfold(X, mode):
    """
    transform tensor X (R^(I1, I2, I3,..., In)) into X^ (R^(Im, I1 x I2 x ... xIn))
    which m = mode

    Author
    Jean Kossaifi <https://github.com/tensorly>
    """
    assert mode < len(X.shape)
    return np.moveaxis(X, mode, 0).reshape((X.shape[mode], -1))


def khatri_rao(matrices):
    """
    Author
    Jean Kossaifi <https://github.com/tensorly>
    """
    n_cols = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i+common_dim for i in target)
    operation = source+'->'+target+common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_cols))


def is_converge(est, data, data_norm, tol, residuals):
    new_resid = np.linalg.norm(est - data)
    residuals.append(new_resid)
    if len(residuals) <= 1:
        return False
    if (abs(residuals[-2] - new_resid) / data_norm) < tol:
        return True
    return False


def rebalance(U_list):
    norms = [np.linalg.norm(U, axis=0) for U in U_list]
    lamb = np.prod(norms, axis=0) ** (1 / len(U_list))
    return [f * (lamb / fn) for f, fn in zip(U_list, norms)]


def cp_als(X,
           n_components = None,
           max_iter=25,
           tol=1e-5):
    norm_X = np.linalg.norm(X)
    modes = X.shape
    unfolded_tensors = [unfold(X, i) for i in range(X.ndim)]
    Us = [np.random.default_rng().standard_normal((modes[i], n_components)) for i in range(X.ndim)]
    res_histories, est = [], None
    for _ in range(max_iter):
        for n in range(X.ndim):
            Us = rebalance(Us)
            components = [Us[j] for j in range(X.ndim) if j != n]
            grams = np.prod([u.T @ u for u in components], axis=0)

            kr = khatri_rao(components)

            c = scipy.linalg.cho_factor(grams, overwrite_a=False)
            p = unfolded_tensors[n].dot(kr)
            Us[n] = scipy.linalg.cho_solve(c, p.T, overwrite_b=False).T
        est = np.reshape(Us[0] @ khatri_rao(Us[1:]).T, tuple([u.shape[0] for u in Us]))
        if is_converge(est=est, data=X, data_norm=norm_X, tol=tol,
                       residuals=res_histories):
            break
    return Us, est, res_histories


@timer
def tensor_decomp(networks,
                  gene_names,
                  n_decimal = 1,
                  K = 5,
                  tol = 1e-5,
                  max_iter=1000,
                  **kwargs) -> pd.DataFrame:
    Us, est, res_hist = cp_als(networks, n_components=K, max_iter=max_iter, tol=tol)
    print(est.shape, len(Us), res_hist)
    out = np.sum(est, axis=-1) / len(networks)
    out = np.round(out / np.max(abs(out)), n_decimal)
    return pd.DataFrame(out, index=gene_names, columns=gene_names)