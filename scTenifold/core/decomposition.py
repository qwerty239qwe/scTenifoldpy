import numpy as np
import pandas as pd
import scipy
from scTenifold.core.utils import timer
from tensorly.decomposition import parafac
import tensorly as tl

# def randomized_range_finder(A, *, size, n_iter,
#                             power_iteration_normalizer='auto',
#                             random_state=None):
#     """Computes an orthonormal matrix whose range approximates the range of A.
#     Parameters
#     ----------
#     A : 2D array
#         The input data matrix.
#     size : int
#         Size of the return array.
#     n_iter : int
#         Number of power iterations used to stabilize the result.
#     power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
#         Whether the power iterations are normalized with step-by-step
#         QR factorization (the slowest but most accurate), 'none'
#         (the fastest but numerically unstable when `n_iter` is large, e.g.
#         typically 5 or larger), or 'LU' factorization (numerically stable
#         but can lose slightly in accuracy). The 'auto' mode applies no
#         normalization if `n_iter` <= 2 and switches to LU otherwise.
#         .. versionadded:: 0.18
#     random_state : int, RandomState instance or None, default=None
#         The seed of the pseudo random number generator to use when shuffling
#         the data, i.e. getting the random vectors to initialize the algorithm.
#         Pass an int for reproducible results across multiple function calls.
#         See :term:`Glossary <random_state>`.
#     Returns
#     -------
#     Q : ndarray
#         A (size x size) projection matrix, the range of which
#         approximates well the range of the input matrix A.
#     Notes
#     -----
#     Follows Algorithm 4.3 of
#     Finding structure with randomness: Stochastic algorithms for constructing
#     approximate matrix decompositions
#     Halko, et al., 2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf
#     An implementation of a randomized algorithm for principal component
#     analysis
#     A. Szlam et al. 2014
#     """
#     random_state = check_random_state(random_state)
#
#     # Generating normal random vectors with shape: (A.shape[1], size)
#     Q = random_state.normal(size=(A.shape[1], size))
#     if A.dtype.kind == 'f':
#         # Ensure f32 is preserved as f32
#         Q = Q.astype(A.dtype, copy=False)
#
#     # Deal with "auto" mode
#     if power_iteration_normalizer == 'auto':
#         if n_iter <= 2:
#             power_iteration_normalizer = 'none'
#         else:
#             power_iteration_normalizer = 'LU'
#
#     # Perform power iterations with Q to further 'imprint' the top
#     # singular vectors of A in Q
#     for i in range(n_iter):
#         if power_iteration_normalizer == 'none':
#             Q = safe_sparse_dot(A, Q)
#             Q = safe_sparse_dot(A.T, Q)
#         elif power_iteration_normalizer == 'LU':
#             Q, _ = np.linalg.lu(safe_sparse_dot(A, Q), permute_l=True)
#             Q, _ = np.linalg.lu(safe_sparse_dot(A.T, Q), permute_l=True)
#         elif power_iteration_normalizer == 'QR':
#             Q, _ = np.linalg.qr(safe_sparse_dot(A, Q), mode='economic')
#             Q, _ = np.linalg.qr(safe_sparse_dot(A.T, Q), mode='economic')
#
#     # Sample the range of A using by linear projection of Q
#     # Extract an orthonormal basis
#     Q, _ = np.linalg.qr(safe_sparse_dot(A, Q), mode='economic')
#     return Q


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
                  tol = 1e-6,
                  max_iter=1000,
                  random_state=42,
                  **kwargs) -> pd.DataFrame:
    # Us, est, res_hist = cp_als(networks, n_components=K, max_iter=max_iter, tol=tol)
    # print(est.shape, len(Us), res_hist)
    print("Using tensorly")
    factors = parafac(networks, rank=K, n_iter_max=max_iter, tol=tol, random_state=random_state)
    estimate = tl.cp_to_tensor(factors)
    print(estimate.shape)
    out = np.sum(estimate, axis=-1) / len(networks)
    out = np.round(out / np.max(abs(out)), n_decimal)
    return pd.DataFrame(out, index=gene_names, columns=gene_names)