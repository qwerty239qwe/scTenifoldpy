from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from scTenifold.core._networks import make_networks


def ko_propagation(B, x, ko_gene_id, degree: int) -> np.ndarray:
    adj_mat = B.copy()
    np.fill_diagonal(adj_mat, 0)
    x_ko = x.copy()
    p = np.zeros(shape=x.shape)
    p[ko_gene_id, :] = x[ko_gene_id, :]
    perturbs = [p]
    is_visited = np.array([False for _ in range(x_ko.shape[0])])
    for d in range(degree):
        if not is_visited.all():
            perturbs.append(adj_mat @ perturbs[d])
            new_visited = (perturbs[d+1] != 0).any(axis=1)
            adj_mat[is_visited, :] = 0
            adj_mat[:, is_visited] = 0
            is_visited = is_visited | new_visited

    for p in perturbs:
        x_ko = x_ko - p
    return np.where(x_ko >= 0, x_ko, 0)


def reconstruct_pcnets(nets: List[coo_matrix],
                       X_df,
                       ko_gene_id,
                       degree,
                       **kwargs):
    ko_nets = []
    for net in nets:
        data = ko_propagation(net.toarray(), X_df.values, ko_gene_id, degree)
        data = pd.DataFrame(data, index=X_df.index, columns=X_df.columns)
        ko_net = make_networks(data, n_nets=1, **kwargs)[0]
        ko_nets.append(ko_net)
    return ko_nets