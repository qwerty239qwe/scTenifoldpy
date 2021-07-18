from dataclasses import dataclass
from typing import Optional, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from anndata import AnnData


__all__ = ("DEFAULT_POS", "DEFAULT_NEG", "TestData")


DEFAULT_POS = ["CD44", "LY6C", "KLRG1", "CTLA", "ICOS", "LAG3"]
DEFAULT_NEG = ["IL2", "TNF"]


def _normalize(data):
    return np.log((data / np.nansum(data, axis=0)) * 1e4 + 1)


@dataclass
class TestData:
    n_genes: int = 1000
    n_samples: int = 100
    pos_eff_ratio: float = 0.3
    neg_eff_ratio: float = 0
    target_pos: Optional[Sequence[str]] = None
    target_neg: Optional[Sequence[str]] = None
    n_bins: int = 25
    n_ctrl: int = 50
    random_state: int = 42

    def __post_init__(self):
        self.random_state_seed = self.random_state
        random_state = np.random.default_rng(self.random_state)
        if self.target_pos is None:
            self.target_pos = DEFAULT_POS
        if self.target_neg is None:
            self.target_neg = []
        self.X = random_state.negative_binomial(20, 0.9,
                                                size=(self.n_genes, self.n_samples))
        self._add_eff(random_state)

        self.gene_list = ([f"pseudo_G{i}" for i in range(self.n_genes -
                                                         len(self.target_pos) -
                                                         len(self.target_neg))] +
                          self.target_pos + self.target_neg)
        self.samples = [f"cell{i}" for i in range(self.X.shape[1])]
        self.n_X = _normalize(self.X)

    def _add_eff(self, random_state):
        pos_eff_size = int(self.n_samples * self.pos_eff_ratio)
        neg_eff_size = int(self.n_samples * self.neg_eff_ratio)
        effect = np.zeros(shape=(self.n_genes, pos_eff_size+neg_eff_size), dtype=np.int32)
        effect[-len(self.target_pos)-len(self.target_neg):
               -len(self.target_neg), :pos_eff_size] = random_state.negative_binomial(20, 0.5,
                                                                                      size=(len(self.target_pos), pos_eff_size))
        effect[-len(self.target_neg):, pos_eff_size:] = random_state.negative_binomial(20, 0.5,
                                                                                       size=(len(self.target_neg), neg_eff_size))
        self.X[:, -pos_eff_size-neg_eff_size:] += effect

    def save_data(self, file_path, use_normalized):
        self.get_data("pandas", use_normalized)["X"].to_csv(file_path)

    def get_data(self, data_type, use_normalized):
        used_X = self.n_X if use_normalized else self.X
        if data_type == "ann_data":
            used_X = AnnData(
                sparse.csr_matrix(used_X.T),
                obs=pd.DataFrame(index=self.samples),
                var=pd.DataFrame(index=self.gene_list),
            )
            return {"random_state": self.random_state_seed,
                    "adata": used_X,
                    "gene_list": self.target_pos,
                    "n_bins": self.n_bins,
                    "ctrl_size": self.n_ctrl,
                    "copy": True}
        elif data_type == "numpy":
            return {"random_state": self.random_state_seed,
                    "X": used_X,
                    "gene_list": self.gene_list,
                    "sample_list": self.samples,
                    "n_bins": self.n_bins,
                    "n_ctrl": self.n_ctrl}
        elif data_type == "pandas":
            used_X = pd.DataFrame(used_X,
                                  index=self.gene_list,
                                  columns=self.samples)
            return {"random_state": self.random_state_seed,
                    "X": used_X,
                    "genes": self.target_pos,
                    "n_bins": self.n_bins,
                    "n_ctrl": self.n_ctrl}
