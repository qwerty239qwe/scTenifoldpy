from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from scTenifold.core._networks import make_networks


def ko_propagation(B, x, ko_gene_id, degree: int) -> np.ndarray:
    adj_mat = B  # toarray() in the caller already returns a fresh array
    np.fill_diagonal(adj_mat, 0)
    x_ko = x.astype(float)
    p0 = np.zeros(shape=x.shape)
    p0[ko_gene_id, :] = x[ko_gene_id, :]
    is_visited = np.zeros(x_ko.shape[0], dtype=bool)
    x_ko -= p0
    current = p0
    for _ in range(degree):
        if not is_visited.all():
            current = adj_mat @ current
            new_visited = (current != 0).any(axis=1)
            adj_mat[is_visited, :] = 0
            adj_mat[:, is_visited] = 0
            is_visited |= new_visited
            x_ko -= current
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