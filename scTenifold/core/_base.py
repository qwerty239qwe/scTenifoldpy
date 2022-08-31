import json
import time
from pathlib import Path
from typing import Optional, Union
import inspect

import numpy as np
import pandas as pd
from scipy import sparse

from scTenifold.core._networks import *
from scTenifold.core._QC import sc_QC
from scTenifold.core._norm import cpm_norm
from scTenifold.core._decomposition import tensor_decomp
from scTenifold.core._ko import reconstruct_pcnets
from scTenifold.plotting import plot_hist
from scTenifold.data import read_folder

__all__ = ["scTenifoldNet", "scTenifoldKnk"]


class scBase:
    cls_prop = ["shared_gene_names", "strict_lambda"]
    kw_sigs = {"qc_kws": inspect.signature(sc_QC),
               "nc_kws": inspect.signature(make_networks),
               "td_kws": inspect.signature(tensor_decomp),
               "ma_kws": inspect.signature(manifold_alignment),
               "dr_kws": inspect.signature(d_regulation)}

    def __init__(self,
                 qc_kws=None,
                 nc_kws=None,
                 td_kws=None,
                 ma_kws=None,
                 dr_kws=None
                 ):
        self.data_dict = {}
        self.QC_dict = {}
        self.network_dict = {}
        self.tensor_dict = {}
        self.manifold: Optional[pd.DataFrame] = None
        self.d_regulation: Optional[pd.DataFrame] = None
        self.shared_gene_names = None
        self.qc_kws = {} if qc_kws is None else qc_kws
        self.nc_kws = {} if nc_kws is None else nc_kws
        self.td_kws = {} if td_kws is None else td_kws
        self.ma_kws = {} if ma_kws is None else ma_kws
        self.dr_kws = {} if dr_kws is None else dr_kws
        self.step_comps = {"qc": self.QC_dict,
                           "nc": self.network_dict,
                           "td": self.tensor_dict,
                           "ma": self.manifold,
                           "dr": self.d_regulation}

    @classmethod
    def _load_comp(cls,
                   file_dir: Path,
                   comp):
        if comp == "qc":
            dic = {}
            for d in file_dir.iterdir():
                if d.is_file():
                    dic[d.stem] = pd.read_csv(d)
            obj_name = "QC_dict"
        elif comp == "nc":
            dic = {}
            for d in file_dir.iterdir():
                if d.is_dir():
                    dic[d.stem] = []
                    nt = 0
                    while (d / Path(f"network_{nt}.npz")).exists():
                        dic[d.stem].append(sparse.load_npz(d / Path(f"network_{nt}.npz")))
                        nt += 1
            obj_name = "network_dict"
        elif comp == "td":
            dic = {}
            for d in file_dir.iterdir():
                if d.is_file():
                    dic[d.stem] = sparse.load_npz(d).toarray()
            obj_name = "tensor_dict"
        elif comp in ["ma", "dr"]:
            dic = {}
            for d in file_dir.iterdir():
                if d.is_file():
                    dic[d.stem] = pd.read_csv(d)
            obj_name = "manifold" if comp == "ma" else "d_regulation"
        else:
            raise ValueError("The component is not a valid one")
        return dic, obj_name

    @classmethod
    def load(cls,
             file_dir,
             **kwargs):
        parent_dir = Path(file_dir)
        kw_path = parent_dir / Path("kws.json")
        with open(kw_path, "r") as f:
            kws = json.load(f)
        kwargs.update(kws)
        kwarg_props = {k: kwargs.pop(k)
                       for k in cls.cls_prop if k in kwargs}
        ins = cls(**kwargs)
        for name, obj in ins.step_comps.items():
            if (file_dir / Path(name)).exists():
                dic, name = cls._load_comp(file_dir / Path(name), name)
                setattr(ins, name, dic)
        for k, prop in kwarg_props.items():
            setattr(ins, k, prop)
        return ins

    @classmethod
    def list_kws(cls, step_name):
        return {n: p.default for n, p in cls.kw_sigs[f"{step_name}"].parameters.items()
                if not (p.default is p.empty)}

    @staticmethod
    def _infer_groups(*args):
        grps = set()
        for kw in args:
            grps |= set(kw.keys())
        return list(grps)

    def _QC(self, label, plot: bool = True, **kwargs):
        self.QC_dict[label] = self.data_dict[label].copy()
        self.QC_dict[label].loc[:, "gene"] = self.QC_dict[label].index
        self.QC_dict[label] = self.QC_dict[label].groupby(by="gene").sum()
        self.QC_dict[label] = sc_QC(self.QC_dict[label], **kwargs)
        if plot:
            plot_hist(self.QC_dict[label], label)

    def _make_networks(self, label, data, **kwargs):
        self.network_dict[label] = make_networks(data, **kwargs)

    def _tensor_decomp(self, label, gene_names, **kwargs):
        self.tensor_dict[label] = tensor_decomp(np.concatenate([np.expand_dims(network.toarray(), -1)
                                                                for network in self.network_dict[label]], axis=-1),
                                                gene_names, **kwargs)

    def _save_comp(self,
                   file_dir: Path,
                   comp: str,
                   verbose: bool):
        if comp == "qc":
            for label, obj in self.step_comps["qc"].items():
                label_fn = (file_dir / Path(label)).with_suffix(".csv")
                obj.to_csv(label_fn)
                if verbose:
                    print(f"{label_fn.name} has been saved successfully.")
        elif comp == "nc":
            for label, obj in self.step_comps["nc"].items():
                (file_dir / Path(f"{label}")).mkdir(parents=True, exist_ok=True)
                for i, npx in enumerate(obj):
                    file_name = file_dir / Path(f"{label}/network_{i}").with_suffix(".npz")
                    sparse.save_npz(file_name, npx)
                    if verbose:
                        print(f"{file_name.name} has been saved successfully.")
        elif comp == "td":
            for label, obj in self.step_comps["td"].items():
                sp = sparse.coo_matrix(obj)
                label_fn = (file_dir / Path(label)).with_suffix(".npz")
                sparse.save_npz(label_fn, sp)
                if verbose:
                    print(f"{label_fn.name} has been saved successfully.")
        elif comp in ["ma", "td"]:
            if isinstance(self.step_comps["ma"], pd.DataFrame):
                fn = (file_dir / Path("manifold_alignment" if comp == "ma" else "d_regulation")).with_suffix(".csv")
                self.step_comps[comp].to_csv(fn)
                if verbose:
                    print(f"{fn.name} has been saved successfully.")
        else:
            raise ValueError(f"This step is not valid, please choose from {list(self.step_comps.keys())}")

    def save(self,
             file_dir: str,
             comps: Union[str, list] = "all",
             verbose: bool = True,
             **kwargs):
        dir_path = Path(file_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        if comps == "all":
            comps = [k for k, v in self.step_comps.items()
                     if v is not None or (isinstance(v, dict) and len(v) != 0)]
        for c in comps:
            subdir = dir_path / Path(c)
            subdir.mkdir(parents=True, exist_ok=True)
            self._save_comp(subdir, c, verbose)
        configs = {"qc_kws": self.qc_kws, "nc_kws": self.nc_kws, "td_kws": self.td_kws, "ma_kws": self.ma_kws}
        if hasattr(self, "ko_kws"):
            configs.update({"ko_kws": getattr(self, "ko_kws")})
        if hasattr(self, "dr_kws"):
            configs.update({"dr_kws": getattr(self, "dr_kws")})
        if self.shared_gene_names is not None:
            configs.update({"shared_gene_names": self.shared_gene_names})
        configs.update(kwargs)
        with open(dir_path / Path('kws.json'), 'w') as f:
            json.dump(configs, f)


class scTenifoldNet(scBase):
    def __init__(self,
                 x_data: pd.DataFrame,
                 y_data: pd.DataFrame,
                 x_label: str,
                 y_label: str,
                 qc_kws: dict = None,
                 nc_kws: dict = None,
                 td_kws: dict = None,
                 ma_kws: dict = None,
                 dr_kws: dict = None):
        """

        Parameters
        ----------
        x_data: pd.DataFrame
            DataFrame contains single-cell data (rows: genes, cols: cells)
        y_data: pd.DataFrame
            DataFrame contains single-cell data (rows: genes, cols: cells)
        x_label: str
            The label of x_data
        y_label: str
            The label of y_data
        qc_kws: dict
            Keyword arguments of the QC step
        nc_kws: dict
            Keyword arguments of the network constructing step
        td_kws: dict
            Keyword arguments of the tensor decomposition step
        ma_kws: dict
            Keyword arguments of the manifold alignment step
        """
        super().__init__(qc_kws=qc_kws, nc_kws=nc_kws, td_kws=td_kws, ma_kws=ma_kws, dr_kws=dr_kws)
        self.x_label, self.y_label = x_label, y_label
        self.data_dict[x_label] = x_data
        self.data_dict[y_label] = y_data

    @classmethod
    def get_empty_config(cls):
        config = {"x_data_path": None, "y_data_path": None,
                  "x_label": None, "y_label": None}
        for kw, sig in cls.kw_sigs.items():
            config[kw] = cls.list_kws(kw)
        return config

    @classmethod
    def load_config(cls, config):
        x_data_path = Path(config.pop("x_data_path"))
        y_data_path = Path(config.pop("y_data_path"))
        if x_data_path.is_dir():
            x_data = read_folder(x_data_path)
        else:
            x_data = pd.read_csv(x_data_path, sep='\t' if x_data_path.suffix == ".tsv" else ",")
        if y_data_path.is_dir():
            y_data = read_folder(y_data_path)
        else:
            y_data = pd.read_csv(y_data_path, sep='\t' if y_data_path.suffix == ".tsv" else ",")
        return cls(x_data, y_data, **config)

    def save(self,
             file_dir: str,
             comps: Union[str, list] = "all",
             verbose: bool = True,
             **kwargs):
        super().save(file_dir, comps, verbose,
                     x_data="", y_data="",    # TODO: fix this later
                     x_label=self.x_label, y_label=self.y_label)

    def _norm(self, label):
        self.QC_dict[label] = cpm_norm(self.QC_dict[label])

    def run_step(self,
                 step_name: str,
                 **kwargs) -> None:
        """
        Run a single step of scTenifoldNet

        Parameters
        ----------
        step_name: str
            The name of step to be run, possible steps:
            1. qc: Quality control
            2. nc: Network construction (PCNet)
            3. td: Tensor decomposition
            4. ma: Manifold alignment
            5. dr: Differential regulation evaluation
        **kwargs
            Keyword arguments for the step, if None then use stored kws in this object.

        Returns
        -------
        None
        """
        start_time = time.perf_counter()
        if step_name == "qc":
            for label in self.data_dict:
                self._QC(label,
                         **(self.qc_kws if kwargs == {} else kwargs))
                self._norm(label)
                print("finish QC:", label)
        elif step_name == "nc":
            x_gene_names, y_gene_names = set(self.QC_dict[self.x_label].index), set(self.QC_dict[self.y_label].index)
            self.shared_gene_names = list(x_gene_names & y_gene_names)
            for label, qc_data in self.QC_dict.items():
                self._make_networks(label, data=qc_data.loc[self.shared_gene_names, :],
                                    **(self.nc_kws if kwargs == {} else kwargs))
        elif step_name == "td":
            for label, qc_data in self.QC_dict.items():
                self._tensor_decomp(label, self.shared_gene_names, **(self.td_kws if kwargs == {} else kwargs))
            self.tensor_dict[self.x_label] = (self.tensor_dict[self.x_label] + self.tensor_dict[self.x_label].T) / 2
            self.tensor_dict[self.y_label] = (self.tensor_dict[self.y_label] + self.tensor_dict[self.y_label].T) / 2
        elif step_name == "ma":
            self.manifold = manifold_alignment(self.tensor_dict[self.x_label],
                                               self.tensor_dict[self.y_label],
                                               **(self.ma_kws if kwargs == {} else kwargs))
        elif step_name == "dr":
            self.d_regulation = d_regulation(self.manifold, **(self.dr_kws if kwargs == {} else kwargs))
        else:
            raise ValueError("This step name is not valid, please choose from qc, nc, td, ma, dr")

        print(f"process {step_name} finished in {time.perf_counter() - start_time} secs.")

    def build(self) -> pd.DataFrame:
        """
        Run the whole pipeline of scTenifoldNet

        Returns
        -------
        d_regulation_df: pd.DataFrame
            Differential regulation result dataframe
        """
        self.run_step("qc")
        self.run_step("nc")
        self.run_step("td")
        self.run_step("ma")
        self.run_step("dr")
        return self.d_regulation


class scTenifoldKnk(scBase):
    def __init__(self,
                 data,
                 strict_lambda=0,
                 ko_method="default",
                 ko_genes=None,
                 qc_kws=None,
                 nc_kws=None,
                 td_kws=None,
                 ma_kws=None,
                 dr_kws=None,
                 ko_kws=None):
        """

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame contains single-cell data (rows: genes, cols: cells)
        strict_lambda: float
            strict_direction's parameter, default: 0
        ko_method: str
            KO method, ['default', 'propagation']
        ko_genes: str, list of str
            Gene(s) to be knocked out
        qc_kws: dict
            Keyword arguments of the QC step
        nc_kws: dict
            Keyword arguments of the network constructing step
        td_kws: dict
            Keyword arguments of the tensor decomposition step
        ma_kws: dict
            Keyword arguments of the manifold alignment step
        ko_kws: dict
            Keyword arguments of the Knock out step
        """
        ma_kws = {"d": 2} if ma_kws is None else ma_kws
        super().__init__(qc_kws=qc_kws, nc_kws=nc_kws, td_kws=td_kws, ma_kws=ma_kws, dr_kws=dr_kws)
        self.data_dict["WT"] = data
        self.strict_lambda = strict_lambda
        self.ko_genes = ko_genes if ko_genes is not None else []
        self.ko_method = ko_method
        self.ko_kws = {} if ko_kws is None else ko_kws

    @classmethod
    def get_empty_config(cls):
        config = {"data_path": None, "strict_lambda": 0,
                  "ko_method": "default", "ko_genes": []}
        for kw, sig in cls.kw_sigs.items():
            config[kw] = cls.list_kws(kw)
        return config

    @classmethod
    def load_config(cls, config):
        data_path = Path(config.pop("data_path"))
        if data_path.is_dir():
            data = read_folder(data_path)
        else:
            data = pd.read_csv(data_path, sep='\t' if data_path.suffix == ".tsv" else ",")
        return cls(data, **config)

    def save(self,
             file_dir: str,
             comps: Union[str, list] = "all",
             verbose: bool = True,
             **kwargs):
        super().save(file_dir, comps, verbose,
                     data="",  # TODO: fix this later
                     ko_method=self.ko_method,
                     strict_lambda=self.strict_lambda, ko_genes=self.ko_genes)

    def _get_ko_tensor(self, ko_genes, **kwargs):
        if self.ko_method == "default":
            self.tensor_dict["KO"] = self.tensor_dict["WT"].copy()
            self.tensor_dict["KO"].loc[ko_genes, :] = 0
        elif self.ko_method == "propagation":
            print(self.QC_dict["WT"].index)
            self.network_dict["KO"] = reconstruct_pcnets(self.network_dict["WT"],
                                                         self.QC_dict["WT"],
                                                         ko_gene_id=[self.QC_dict["WT"].index.get_loc(i)
                                                                     for i in ko_genes],
                                                         degree=kwargs.get("degree"),
                                                         **self.nc_kws)
            self._tensor_decomp("KO", self.shared_gene_names, **self.td_kws)
            self.tensor_dict["KO"] = strict_direction(self.tensor_dict["KO"], self.strict_lambda).T
            np.fill_diagonal(self.tensor_dict["KO"].values, 0)
        else:
            ValueError("No such method")

    def run_step(self,
                 step_name: str,
                 **kwargs):
        """
        Run a single step of scTenifoldKnk

        Parameters
        ----------
        step_name: str
            The name of step to be run, possible steps:
            1. qc: Quality control
            2. nc: Network construction (PCNet)
            3. td: Tensor decomposition
            4. ko: Virtual knock out
            5. ma: Manifold alignment
            6. dr: Differential regulation evaluation
        **kwargs
            Keyword arguments for the step, if None then use stored kws in this object.

        Returns
        -------
        None
        """
        start_time = time.perf_counter()
        if step_name == "qc":
            if "min_exp_avg" not in self.qc_kws:
                self.qc_kws["min_exp_avg"] = 0.05
            if "min_exp_sum" not in self.qc_kws:
                self.qc_kws["min_exp_sum"] = 25
            self._QC("WT", **(self.qc_kws if kwargs == {} else kwargs))
            # no norm
            print("finish QC: WT")
        elif step_name == "nc":
            self._make_networks("WT", self.QC_dict["WT"], **(self.nc_kws if kwargs == {} else kwargs))
            self.shared_gene_names = self.QC_dict["WT"].index.to_list()
        elif step_name == "td":
            self._tensor_decomp("WT", self.shared_gene_names, **(self.td_kws if kwargs == {} else kwargs))
            self.tensor_dict["WT"] = strict_direction(self.tensor_dict["WT"], self.strict_lambda).T
        elif step_name == "ko":
            np.fill_diagonal(self.tensor_dict["WT"].values, 0)
            if kwargs.get("ko_genes") is not None:
                ko_genes = kwargs.pop("ko_genes")
                kwargs = (self.ko_kws if kwargs == {} else kwargs)
            else:
                ko_genes = self.ko_genes
                kwargs = (self.ko_kws if kwargs == {} else kwargs)
            self._get_ko_tensor(ko_genes, **kwargs)
        elif step_name == "ma":
            self.manifold = manifold_alignment(self.tensor_dict["WT"],
                                               self.tensor_dict["KO"],
                                               **(self.ma_kws if kwargs == {} else kwargs))
        elif step_name == "dr":
            self.d_regulation = d_regulation(self.manifold, **(self.dr_kws if kwargs == {} else kwargs))
        else:
            raise ValueError("No such step")
        print(f"process {step_name} finished in {time.perf_counter() - start_time} secs.")

    def build(self):
        """
        Run the whole pipeline of scTenifoldKnk

        Returns
        -------
        d_regulation_df: pd.DataFrame
            Differential regulation result dataframe
        """
        self.run_step("qc")
        self.run_step("nc")
        self.run_step("td")
        self.run_step("ko")
        self.run_step("ma")
        self.run_step("dr")
        return self.d_regulation
