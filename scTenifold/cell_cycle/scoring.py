import numpy as np
import pandas as pd
from scipy import sparse
from anndata import AnnData
from pathlib import Path
from scanpy.tools import score_genes
from typing import Optional, Dict, List, Sequence
from dataclasses import dataclass
import argparse


DEFAULT_POS = ["CD44", "LY6C", "KLRG1", "CTLA", "ICOS", "LAG3"]
DEFAULT_NEG = ["IL2", "TNF"]


def _normalize(data):
    return np.log((data / np.nansum(data, axis=0)) * 1e4 + 1)


@dataclass
class TestData:

    n_genes: int = 1000
    n_samples: int = 100
    eff_ratio: float = 0.3
    target_pos: Optional[Sequence[str]] = None
    n_bins: int = 25
    n_ctrl: int = 50
    random_state: int = 42

    def __post_init__(self):
        self.random_state_seed = self.random_state
        random_state = np.random.default_rng(self.random_state)
        if self.target_pos is None:
            self.target_pos = DEFAULT_POS
        self.X = random_state.negative_binomial(20, 0.9,
                                                size=(self.n_genes, self.n_samples))
        eff_size = int(self.n_samples * self.eff_ratio)
        effect = np.zeros(shape=(self.n_genes, eff_size), dtype=np.int32)
        effect[-len(self.target_pos):, :] = random_state.negative_binomial(20, 0.5,
                                                                      size=(len(self.target_pos), eff_size))
        self.X[:, -eff_size:] += effect
        self.gene_list = [f"pseudo_G{i}" for i in range(self.n_genes - len(self.target_pos))] + self.target_pos
        self.samples = [f"cell{i}" for i in range(self.X.shape[1])]
        self.n_X = _normalize(self.X)

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


def adobo_score(X,
                genes,
                n_bins: int = 25,
                n_ctrl: int = 50,
                random_state: int = 42,
                file_path: Path = None):
    if len(genes) == 0:
        raise ValueError('Gene list ("genes") is empty.')
    gene_mean = X.mean(axis=1)
    gene_mean = gene_mean.sort_values()
    binned = pd.qcut(gene_mean, n_bins)
    ret = []
    for g in genes:
        sampled_bin = binned[binned == binned[binned.index == g].values[0]]
        if n_ctrl > sampled_bin.shape[0]:
            ret.append(sampled_bin.index)
        else:
            ret.append(
                sampled_bin.sample(n_ctrl, replace=True, random_state=random_state).index
            )
    con = []
    for g in ret:
        con.append(X[X.index.isin(g)].mean(axis=0))

    con = pd.concat(con, axis=1).transpose()
    con.index = genes
    targets = X[X.index.isin(genes)]
    targets = targets.reindex(genes)
    scores = (targets-con).mean(axis=0)
    if file_path:
        scores.to_csv(file_path)
    return scores


def _get_assigned_bins(data_avg: np.ndarray,
                       cluster_len: int,
                       n_bins: int) -> np.ndarray:
    assigned_bin = np.zeros(shape=(cluster_len, ), dtype=np.int32)  # (G,)
    bin_size = cluster_len / n_bins
    for i_bin in range(n_bins):
        assigned_bin[(assigned_bin == 0) &
                     (data_avg <= data_avg[int(np.round(bin_size * i_bin))])] = i_bin
    return assigned_bin


def _get_ctrl_use(assigned_bin: np.ndarray,
                  gene_arr,
                  target_dict,
                  n_ctrl,
                  random_state) -> List[str]:
    selected_bins = list(set(assigned_bin[np.in1d(gene_arr, target_dict["Pos"])]))
    genes_in_same_bin = gene_arr[np.in1d(assigned_bin, selected_bins)]
    ctrl_use = list()
    for _ in range(len(target_dict["Pos"])):
        ctrl_use.extend(random_state.choice(genes_in_same_bin, n_ctrl))
    return list(set(ctrl_use))


def cell_cycle_score(X,
                     gene_list: List[str],
                     sample_list: List[str],
                     target_dict: Optional[Dict[str, List[str]]] = None,
                     n_bins: int = 25,
                     n_ctrl: int = 50,
                     random_state: int = 42,
                     file_path: Optional[Path] = None):
    random_state = np.random.default_rng(random_state)
    if target_dict is None:
        target_dict = {"Pos": DEFAULT_POS,
                       "Neg": DEFAULT_NEG}
    else:
        target_dict = {k: [i.upper() for i in v] for k, v in target_dict.items()}

    if len(set(gene_list) & set(target_dict["Pos"])) == 0:
        raise ValueError('No feature genes found in gene_list.')

    gene_list = [i.upper() for i in gene_list]
    cluster_len = X.shape[0]
    data_avg = X.mean(axis=1)
    sort_arg = np.argsort(data_avg)
    data_avg = data_avg[sort_arg]
    gene_list = np.array(gene_list)[sort_arg]
    X = X[sort_arg, :]

    assigned_bin = _get_assigned_bins(data_avg, cluster_len, n_bins)
    used_ctrl = _get_ctrl_use(assigned_bin, gene_list, target_dict,
                              n_ctrl, random_state)
    ctrl_score = X[np.in1d(gene_list, used_ctrl), :].mean(axis=0).T
    features_score = X[np.in1d(gene_list, target_dict["Pos"]), :].mean(axis=0).T
    scores = features_score - ctrl_score
    if file_path:
        pd.DataFrame({"score": scores}, index=sample_list).to_csv(file_path)

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--random_state",
                        help="random seed", default=42, type=int)
    parser.add_argument("-o", "--output_path",
                        help="output directory, it will be automatically and recursively created",
                        default=".", type=str)
    parser.add_argument("-g", "--genes",
                        help="number of the genes in the test data",
                        default=1000, type=int)
    parser.add_argument("-s", "--samples",
                        help="number of the samples (cells/observations) in the test data",
                        default=100, type=int)
    parser.add_argument("-b", "--bins",
                        help="number of bins",
                        default=25, type=int)
    parser.add_argument("-c", "--ctrls",
                        help="number of controls",
                        default=50, type=int)
    args = parser.parse_args()

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_obj = TestData(n_genes=args.genes,
                        n_samples=args.samples,
                        n_bins=args.bins,
                        n_ctrl=args.ctrls,
                        random_state=args.random_state)
    data_obj.save_data(output_dir / Path("test_data.csv"), use_normalized=True)
    np_data = data_obj.get_data("numpy", True)
    np_data["file_path"] = output_dir / Path("cell_scores.csv")
    pd_data = data_obj.get_data("pandas", True)
    pd_data["file_path"] = output_dir / Path("adobo_cell_scores.csv")

    cell_cycle_score(**np_data)
    score_genes(**(data_obj.get_data("ann_data", True))).write_csvs(output_dir / Path("scanpy_result"))
    adobo_score(**pd_data)