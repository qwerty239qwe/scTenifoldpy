import logging
import os
import time

import numpy as np
import ray
from scipy import sparse
from scipy.sparse.linalg import svds as sparse_svds
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def pcCoefficients(X, K, nComp):
    y = X[:,K]
    mask = np.ones(X.shape[1], dtype=bool)
    mask[K] = False
    Xi = X[:,mask]
    u, s, VT = sparse_svds(Xi, k=nComp, random_state=0) # Calling Sparse SVD Solver
    #print ('U:', U.shape, 's:', s.shape, 'VT:', VT.shape)
    V = np.fliplr(VT.T)
    #print('V:', V.shape)
    score = Xi @ V
    t = np.linalg.norm(score, axis = 0)
    score_lsq = (score.T / (t**2).reshape(-1,1)).T
    beta = V @ (y.T*score_lsq).flatten()
    return beta.tolist()

def pcNet(X, # X: cell * gene
    nComp: int = 3,
    scale: bool = True,
    symmetric: bool = True,
    q: float = 0., # q: 0-100
    as_sparse: bool = True,
    random_state: int = 0):

    X = X if sparse.issparse(X) else sparse.csr_matrix(X)

    # Standardizing the data
    X = normalize(X, axis=0)

    if nComp < 2 or nComp >= X.shape[1]:
        raise ValueError('nComp should be greater or equal than 2 and lower than the total number of genes')
    else:
        np.random.seed(random_state)
        n = X.shape[1] # genes
        B = np.array([pcCoefficients(X, k, nComp) for k in range(n)])
        A = np.ones((n, n), dtype=float)
        np.fill_diagonal(A, 0)
        for i in range(n):
            A[i, A[i, :]==1] = B[i, :]
        if scale:
            absA = abs(A)
            A = A / np.max(absA)
        if q > 0:
            A[absA < np.percentile(absA, q)] = 0
        if symmetric: # place in the end
            A = (A + A.T)/2

        if as_sparse:
            A = sparse.csc_matrix(A)
        return A

@ray.remote
def pc_net_parallel(X,  # X: cell * gene
                    nComp: int = 3,
                    scale: bool = True,
                    symmetric: bool = True,
                    q: float = 0.,
                    as_sparse: bool = True,
                    random_state: int = 0):
    return pcNet(X, nComp = nComp, scale = scale, symmetric = symmetric, q = q,
                 as_sparse = as_sparse, random_state = random_state)

def pc_net_single(X,  # X: cell * gene
                    nComp: int = 3,
                    scale: bool = True,
                    symmetric: bool = True,
                    q: float = 0.,
                    as_sparse: bool = True,
                    random_state: int = 0):
    return pcNet(X, nComp = nComp, scale = scale, symmetric = symmetric, q = q,
                 as_sparse = as_sparse, random_state = random_state)

def make_pcNet(X,
          nComp: int = 3,
          scale: bool = True,
          symmetric: bool = True,
          q: float = 0.,
          as_sparse: bool = True,
          random_state: int = 0,
          n_cpus: int = 1, # -1: use all CPUs
          timeit: bool = True):
    start_time = time.time()
    if n_cpus != 1:
        if ray.is_initialized():
            ray.shutdown()
        if (n_cpus == -1) or (n_cpus > os.cpu_count()):
            n_cpus = os.cpu_count()
        ray.init(num_cpus = n_cpus)
        logger.info(f'ray init, using {n_cpus} CPUs')

        X_ray = ray.put(X) # put X to distributed object store and return object ref (ID)
        # print(X_ray)
        net = pc_net_parallel.remote(X_ray, nComp = nComp, scale = scale, symmetric = symmetric, q = q,
                    as_sparse = as_sparse, random_state = random_state)
        net = ray.get(net)
    else:
        net = pc_net_single(X, nComp = nComp, scale = scale, symmetric = symmetric, q = q,
            as_sparse = as_sparse, random_state = random_state)
    if ray.is_initialized():
        ray.shutdown()
    if timeit:
        duration = time.time() - start_time
        logger.info(f'execution time of making pcNet: {duration:.2f} s')
    return net

def main():
    counts = np.random.randint(0, 10, (5, 100))
    net = make_pcNet(counts, as_sparse = True, timeit = True, n_cpus = -1)
    logger.info(f'input counts shape: {counts.shape}, make pcNet completed, shape: {net.shape}')

if __name__ == '__main__':
    main()


