import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Union
from warnings import warn
import inspect

import numpy as np
import pandas as pd
from scipy import sparse

from scTenifold.core._networks import *
from scTenifold.core._networks import anndata_to_dataframe
from scTenifold.core._types import ExpressionData, KOMethod, Kwargs
from scTenifold.core._QC import sc_QC
from scTenifold.core._norm import cpm_norm
from scTenifold.core._decomposition import tensor_decomp
from scTenifold.core._ko import reconstruct_pcnets
from scTenifold.plotting import plot_hist
from scTenifold.data import read_folder

__all__ = ["scTenifoldNet", "scTenifoldKnk"]


def _fill_dataframe_diagonal(df: pd.DataFrame, value: float) -> pd.DataFrame:
    data = df.to_numpy(copy=True)
    np.fill_diagonal(data, value)
    return pd.DataFrame(data, index=df.index, columns=df.columns)


class scBase:
    """Shared scaffolding for the scTenifold workflows.

    Holds per-step keyword arguments (``qc_kws``, ``nc_kws``, ``td_kws``,
    ``ma_kws``, ``dr_kws``), the intermediate result dictionaries, and the
    save/load plumbing used by both :class:`scTenifoldNet` and
    :class:`scTenifoldKnk`.
    """

    cls_prop = ["shared_gene_names", "strict_lambda"]
    # Defaults of the R pipelines that differ from the step functions' own
    step_defaults: Dict[str, Kwargs] = {}
    kw_sigs = {"qc_kws": inspect.signature(sc_QC),
               "nc_kws": inspect.signature(make_networks),
               "td_kws": inspect.signature(tensor_decomp),
               "ma_kws": inspect.signature(manifold_alignment),
               "dr_kws": inspect.signature(d_regulation)}

    def __init__(self,
                 qc_kws: Optional[Kwargs] = None,
                 nc_kws: Optional[Kwargs] = None,
                 td_kws: Optional[Kwargs] = None,
                 ma_kws: Optional[Kwargs] = None,
                 dr_kws: Optional[Kwargs] = None,
                 ) -> None:
        """Initialise empty step dicts and bind per-step keyword arguments."""
        self.data_dict = {}
        self.QC_dict = {}
        self.network_dict = {}
        self.tensor_dict = {}
        self.manifold: Optional[pd.DataFrame] = None
        self.d_regulation: Optional[pd.DataFrame] = None
        self.shared_gene_names = None
        self.qc_kws = self._with_defaults("qc_kws", qc_kws)
        self.nc_kws = self._with_defaults("nc_kws", nc_kws)
        self.td_kws = self._with_defaults("td_kws", td_kws)
        self.ma_kws = self._with_defaults("ma_kws", ma_kws)
        self.dr_kws = self._with_defaults("dr_kws", dr_kws)
        self.step_comps = {"qc": self.QC_dict,
                           "nc": self.network_dict,
                           "td": self.tensor_dict,
                           "ma": self.manifold,
                           "dr": self.d_regulation}

    @classmethod
    def _with_defaults(cls, step_name: str, kws: Optional[Kwargs]) -> Kwargs:
        return {**cls.step_defaults.get(step_name, {}), **({} if kws is None else kws)}

    def _step_kws(self, step_name: str, kwargs: Kwargs) -> Kwargs:
        # One-shot overrides replace the stored kws, over the pipeline defaults
        return getattr(self, step_name) if kwargs == {} else self._with_defaults(step_name, kwargs)

    @classmethod
    def _load_comp(cls,
                   file_dir: Path,
                   comp):
        if comp == "qc":
            dic = {}
            for d in file_dir.iterdir():
                if d.is_file():
                    dic[d.stem] = pd.read_csv(d, index_col=0)
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
            dic = None
            for d in file_dir.iterdir():
                if d.is_file():
                    dic = pd.read_csv(d, index_col=0)
                    break
            obj_name = "manifold" if comp == "ma" else "d_regulation"
        else:
            raise ValueError("The component is not a valid one")
        return dic, obj_name

    @classmethod
    def load(cls,
             file_dir: Union[str, Path],
             **kwargs: object) -> "scBase":
        """Reconstruct an instance previously written by :meth:`save`.

        Parameters
        ----------
        file_dir
            Directory containing the ``kws.json`` and per-step subfolders.
        **kwargs
            Extra constructor kwargs that override values loaded from disk.

        Returns
        -------
        Fully populated subclass instance.
        """
        parent_dir = Path(file_dir)
        kw_path = parent_dir / Path("kws.json")
        with open(kw_path, "r") as f:
            kws = json.load(f)
        kwargs.update(kws)
        kwarg_props = {k: kwargs.pop(k)
                       for k in cls.cls_prop if k in kwargs}
        ins = cls(**kwargs)
        for name, obj in ins.step_comps.items():
            if (parent_dir / Path(name)).exists():
                dic, name = cls._load_comp(parent_dir / Path(name), name)
                setattr(ins, name, dic)
        ins.step_comps = {"qc": ins.QC_dict,
                          "nc": ins.network_dict,
                          "td": ins.tensor_dict,
                          "ma": ins.manifold,
                          "dr": ins.d_regulation}
        for k, prop in kwarg_props.items():
            setattr(ins, k, prop)
        return ins

    @classmethod
    def list_kws(cls, step_name: str) -> Kwargs:
        """Return the default keyword arguments for the named pipeline step.

        Parameters
        ----------
        step_name
            One of ``"qc_kws"``, ``"nc_kws"``, ``"td_kws"``, ``"ma_kws"``,
            ``"dr_kws"``.

        Returns
        -------
        Mapping of keyword name to default value.
        """
        kws = {n: p.default for n, p in cls.kw_sigs[f"{step_name}"].parameters.items()
               if not (p.default is p.empty)}
        kws.update(cls.step_defaults.get(step_name, {}))
        return kws

    @staticmethod
    def _infer_groups(*args: Kwargs) -> List[str]:
        grps = set()
        for kw in args:
            grps |= set(kw.keys())
        return list(grps)

    def _QC(self, label, plot: bool = True, **kwargs):
        self.QC_dict[label] = self.data_dict[label].copy()
        self.QC_dict[label].loc[:, "gene"] = self.QC_dict[label].index
        # sort=False keeps the input gene order, which the random
        # initialization of the tensor decomposition depends on
        self.QC_dict[label] = self.QC_dict[label].groupby(by="gene", sort=False).sum()
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
        elif comp in ["ma", "dr"]:
            if isinstance(self.step_comps[comp], pd.DataFrame):
                fn = (file_dir / Path("manifold_alignment" if comp == "ma" else "d_regulation")).with_suffix(".csv")
                self.step_comps[comp].to_csv(fn)
                if verbose:
                    print(f"{fn.name} has been saved successfully.")
        else:
            raise ValueError(f"This step is not valid, please choose from {list(self.step_comps.keys())}")

    def save(self,
             file_dir: Union[str, Path],
             comps: Union[str, List[str]] = "all",
             verbose: bool = True,
             **kwargs: object) -> None:
        """Persist intermediate results and the active kw config to disk.

        Parameters
        ----------
        file_dir
            Output directory; created if missing.
        comps
            ``"all"`` (default) saves every populated step; otherwise pass a
            list of step keys (``"qc"``, ``"nc"``, ``"td"``, ``"ma"``,
            ``"dr"``).
        verbose
            If True, print a line per file written.
        **kwargs
            Extra config entries merged into ``kws.json``.
        """
        dir_path = Path(file_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        if comps == "all":
            comps = [k for k, v in self.step_comps.items()
                     if v is not None and (not isinstance(v, dict) or len(v) != 0)]
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
    """Two-sample scTenifoldNet workflow.

    Pipeline order: ``qc`` → ``nc`` (PC network construction) → ``td``
    (tensor decomposition) → ``ma`` (manifold alignment) → ``dr``
    (differential regulation). Each step persists its output on the
    instance so it can be inspected, saved, or rerun individually via
    :meth:`run_step`. :meth:`build` runs the full pipeline and returns
    the differential regulation table.

    Parameters
    ----------
    x_data, y_data
        Genes-by-cells expression DataFrames (or AnnData-like objects;
        converted via :func:`anndata_to_dataframe`). The two conditions
        being compared.
    x_label, y_label
        Short labels used as keys in ``data_dict``/``QC_dict``/...
        and to identify each condition in saved output.
    qc_kws
        Overrides for :func:`sc_QC` during the QC step.
    nc_kws
        Overrides for :func:`make_networks` during PC network
        construction. Use this dict to pass ``backend``, ``n_jobs``,
        ``random_state``, etc.
    td_kws
        Overrides for :func:`tensor_decomp`.
    ma_kws
        Overrides for :func:`manifold_alignment`.
    dr_kws
        Overrides for :func:`d_regulation`.

    Notes
    -----
    With the default settings the results are the same as those of
    ``scTenifoldNet()`` in R with its default settings and ``seed = 1``.
    """

    step_defaults = {"td_kws": {"K": 3, "n_decimal": 1}}

    def __init__(self,
                 x_data: ExpressionData,
                 y_data: ExpressionData,
                 x_label: str,
                 y_label: str,
                 qc_kws: Optional[Kwargs] = None,
                 nc_kws: Optional[Kwargs] = None,
                 td_kws: Optional[Kwargs] = None,
                 ma_kws: Optional[Kwargs] = None,
                 dr_kws: Optional[Kwargs] = None) -> None:
        """See class docstring for parameter descriptions."""
        super().__init__(qc_kws=qc_kws, nc_kws=nc_kws, td_kws=td_kws, ma_kws=ma_kws, dr_kws=dr_kws)
        self.x_label, self.y_label = x_label, y_label
        self.data_dict[x_label] = pd.DataFrame() if isinstance(x_data, str) and x_data == "" else anndata_to_dataframe(x_data)
        self.data_dict[y_label] = pd.DataFrame() if isinstance(y_data, str) and y_data == "" else anndata_to_dataframe(y_data)

    @classmethod
    def get_empty_config(cls) -> Dict[str, object]:
        """Return a blank scTenifoldNet config dict populated with step defaults."""
        config = {"x_data_path": None, "y_data_path": None,
                  "x_label": None, "y_label": None}
        for kw, sig in cls.kw_sigs.items():
            config[kw] = cls.list_kws(kw)
        return config

    @classmethod
    def load_config(cls, config: Dict[str, object]) -> "scTenifoldNet":
        """Build a scTenifoldNet from a config dict, reading data from disk.

        ``x_data_path`` and ``y_data_path`` may be either a 10x folder
        (loaded via :func:`read_folder`) or a CSV/TSV file.
        """
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
             file_dir: Union[str, Path],
             comps: Union[str, List[str]] = "all",
             verbose: bool = True,
             **kwargs: object) -> None:
        """Save state plus ``x_label``/``y_label`` so :meth:`load` can rebuild."""
        super().save(file_dir, comps, verbose,
                     x_data="", y_data="",
                     x_label=self.x_label, y_label=self.y_label)

    def _norm(self, label):
        self.QC_dict[label] = cpm_norm(self.QC_dict[label])

    def run_step(self,
                 step_name: Literal["qc", "nc", "td", "ma", "dr"],
                 **kwargs: object) -> None:
        """Run a single step of the scTenifoldNet pipeline.

        Steps must be invoked in order — each depends on the state
        produced by the previous one.

        Parameters
        ----------
        step_name
            Which step to run. One of:

            - ``"qc"`` — quality control + CPM normalisation on both
              conditions. Reads ``self.data_dict``; writes
              ``self.QC_dict``.
            - ``"nc"`` — PC network construction on the shared gene
              set. Reads ``self.QC_dict``; writes ``self.network_dict``
              and ``self.shared_gene_names``.
            - ``"td"`` — tensor decomposition. Reads
              ``self.network_dict``; writes ``self.tensor_dict``. The
              self-loops are kept, as the alignment uses them; the R
              package sets them to zero in the networks it returns.
            - ``"ma"`` — manifold alignment between the two decomposed
              tensors, after symmetrising them. Writes ``self.manifold``.
            - ``"dr"`` — differential regulation from the aligned
              manifold. Writes ``self.d_regulation``.
        **kwargs
            One-shot overrides for the step. When non-empty these
            replace the dict stored on the instance (``qc_kws``,
            ``nc_kws``, etc.) for this call only.

        Raises
        ------
        ValueError
            If ``step_name`` is not one of the five values above.
        """
        start_time = time.perf_counter()
        if step_name == "qc":
            for label in self.data_dict:
                self._QC(label, **self._step_kws("qc_kws", kwargs))
                self._norm(label)
                print("finish QC:", label)
        elif step_name == "nc":
            y_gene_names = set(self.QC_dict[self.y_label].index)
            self.shared_gene_names = [gene for gene in self.QC_dict[self.x_label].index if gene in y_gene_names]
            for label, qc_data in self.QC_dict.items():
                self._make_networks(label, data=qc_data.loc[self.shared_gene_names, :],
                                    **self._step_kws("nc_kws", kwargs))
        elif step_name == "td":
            for label, qc_data in self.QC_dict.items():
                self._tensor_decomp(label, self.shared_gene_names, **self._step_kws("td_kws", kwargs))
        elif step_name == "ma":
            x_net, y_net = self.tensor_dict[self.x_label], self.tensor_dict[self.y_label]
            self.manifold = manifold_alignment((x_net + x_net.T) / 2,
                                               (y_net + y_net.T) / 2,
                                               **self._step_kws("ma_kws", kwargs))
            self.step_comps["ma"] = self.manifold
        elif step_name == "dr":
            self.d_regulation = d_regulation(self.manifold, **self._step_kws("dr_kws", kwargs))
            self.step_comps["dr"] = self.d_regulation
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
    """Single-sample scTenifoldKnk virtual-knockout workflow.

    Pipeline order: ``qc`` → ``nc`` → ``td`` → ``ko`` → ``ma`` → ``dr``.
    A wild-type PC network is built from ``data``; the ``ko`` step
    produces a perturbed tensor and the remaining steps compare WT vs.
    KO. Run end-to-end with :meth:`build`, or step-wise with
    :meth:`run_step`.

    Parameters
    ----------
    data
        Genes-by-cells expression matrix (``pandas.DataFrame`` or
        AnnData-like).
    strict_lambda
        Pruning strength forwarded to :func:`strict_direction` when
        post-processing the decomposed WT tensor. ``0`` disables
        pruning.
    ko_method
        How the KO tensor is generated:

        - ``"default"`` — zero out the WT tensor rows for ``ko_genes``.
        - ``"propagation"`` — rebuild PC networks with the targeted
          gene columns masked using :func:`reconstruct_pcnets`, then
          re-decompose.
    ko_genes
        Gene name or iterable of names to knock out. ``None`` stores
        an empty list.
    qc_kws
        Overrides for :func:`sc_QC`.
    nc_kws
        Overrides for :func:`make_networks` (``backend``, ``n_jobs``,
        etc.).
    td_kws
        Overrides for :func:`tensor_decomp`.
    ma_kws
        Overrides for :func:`manifold_alignment`. ``d`` defaults to 2.
    dr_kws
        Overrides for :func:`d_regulation`.
    ko_kws
        Extra kwargs forwarded to the KO step (e.g. ``degree`` for the
        propagation method).

    Notes
    -----
    With the default settings and ``ko_method="default"`` the results are
    the same as those of ``scTenifoldKnk()`` in R with its default settings
    and ``seed = 1``.
    """

    step_defaults = {"nc_kws": {"q": 0.9},
                     "td_kws": {"K": 3, "n_decimal": 3},
                     "ma_kws": {"d": 2}}

    def __init__(self,
                 data: ExpressionData,
                 strict_lambda: float = 0,
                 ko_method: KOMethod = "default",
                 ko_genes: Optional[Union[str, Iterable[str]]] = None,
                 qc_kws: Optional[Kwargs] = None,
                 nc_kws: Optional[Kwargs] = None,
                 td_kws: Optional[Kwargs] = None,
                 ma_kws: Optional[Kwargs] = None,
                 dr_kws: Optional[Kwargs] = None,
                 ko_kws: Optional[Kwargs] = None) -> None:
        """See class docstring for parameter descriptions."""
        super().__init__(qc_kws=qc_kws, nc_kws=nc_kws, td_kws=td_kws, ma_kws=ma_kws, dr_kws=dr_kws)
        self.data_dict["WT"] = pd.DataFrame() if isinstance(data, str) and data == "" else anndata_to_dataframe(data)
        self.strict_lambda = strict_lambda
        self.ko_genes = ko_genes if ko_genes is not None else []
        self.ko_method = ko_method
        self.ko_kws = {} if ko_kws is None else ko_kws

    @classmethod
    def get_empty_config(cls) -> Dict[str, object]:
        """Return a blank scTenifoldKnk config dict populated with step defaults."""
        config = {"data_path": None, "strict_lambda": 0,
                  "ko_method": "default", "ko_genes": []}
        for kw, sig in cls.kw_sigs.items():
            config[kw] = cls.list_kws(kw)
        return config

    @classmethod
    def load_config(cls, config: Dict[str, object]) -> "scTenifoldKnk":
        """Build a scTenifoldKnk from a config dict, reading data from disk."""
        data_path = Path(config.pop("data_path"))
        if data_path.is_dir():
            data = read_folder(data_path)
        else:
            data = pd.read_csv(data_path, sep='\t' if data_path.suffix == ".tsv" else ",")
        return cls(data, **config)

    def save(self,
             file_dir: Union[str, Path],
             comps: Union[str, List[str]] = "all",
             verbose: bool = True,
             **kwargs: object) -> None:
        """Save state plus KO-specific fields so :meth:`load` can rebuild."""
        super().save(file_dir, comps, verbose,
                     data="",
                     ko_method=self.ko_method,
                     strict_lambda=self.strict_lambda, ko_genes=self.ko_genes)

    def _get_ko_tensor(self, ko_genes, **kwargs):
        if self.ko_method not in ("default", "propagation"):
            raise ValueError("No such method")

        # A bare gene name is a valid ko_genes value (see the type hint),
        # but every check/index below assumes an iterable of names; left
        # as-is, a string would be iterated character by character.
        if isinstance(ko_genes, str):
            ko_genes = [ko_genes]

        # Both branches below index into the post-QC gene set
        # (self.tensor_dict["WT"], built from self.shared_gene_names ==
        # self.QC_dict["WT"].index) rather than the raw input. A gene can be
        # a perfectly valid column in the original data and still be missing
        # here if QC (min_lib_size/min_percent/min_exp_avg/min_exp_sum)
        # dropped it — e.g. a knockout target expressed too rarely to pass
        # min_percent. Left unchecked, that reaches pandas as a bare
        # "None of [Index([...])] are in the [index]" KeyError with no
        # indication of why; check up front and say so plainly, split by
        # whether the gene was ever in the input at all.
        available = set(self.tensor_dict["WT"].index)
        missing = [g for g in ko_genes if g not in available]
        if missing:
            raw_genes = set(self.data_dict["WT"].index.astype(str))
            removed_by_qc = [g for g in missing if g in raw_genes]
            unknown = [g for g in missing if g not in raw_genes]
            parts = []
            if removed_by_qc:
                parts.append(f"removed by QC filtering before the knockout step: {removed_by_qc}")
            if unknown:
                parts.append(f"not found in the input data: {unknown}")
            raise ValueError("knockout gene(s) " + "; ".join(parts))

        if self.ko_method == "default":
            self.tensor_dict["KO"] = self.tensor_dict["WT"].copy()
            self.tensor_dict["KO"].loc[ko_genes, :] = 0
            self._warn_no_outgoing_edges(ko_genes)
        elif self.ko_method == "propagation":
            print(self.QC_dict["WT"].index)
            self.network_dict["KO"] = reconstruct_pcnets(self.network_dict["WT"],
                                                         self.QC_dict["WT"],
                                                         ko_gene_id=[self.QC_dict["WT"].index.get_loc(i)
                                                                     for i in ko_genes],
                                                         degree=kwargs.get("degree", 1),
                                                         **self.nc_kws)
            self._tensor_decomp("KO", self.shared_gene_names, **self.td_kws)
            self.tensor_dict["KO"] = strict_direction(self.tensor_dict["KO"], self.strict_lambda).T.copy()
            self.tensor_dict["KO"] = _fill_dataframe_diagonal(self.tensor_dict["KO"], 0)

    def _warn_no_outgoing_edges(self, ko_genes):
        # A gene without outgoing edges leaves the network unchanged
        ko_genes = list(dict.fromkeys(ko_genes))
        no_edges = [g for g in ko_genes if not (self.tensor_dict["WT"].loc[g, :] != 0).any()]
        if ko_genes and len(no_edges) == len(ko_genes):
            warn(f"{', '.join(ko_genes)} {'has' if len(ko_genes) == 1 else 'have'} no outgoing edges in the "
                 "WT network; the knockout does not change the network and the differential regulation "
                 "results reflect only numerical noise.")
        elif no_edges:
            warn("The following genes have no outgoing edges in the WT network, so knocking them out "
                 f"has no effect: {', '.join(no_edges)}")

    def run_step(self,
                 step_name: Literal["qc", "nc", "td", "ko", "ma", "dr"],
                 **kwargs: object) -> None:
        """Run a single step of the scTenifoldKnk pipeline.

        Steps must be invoked in order — each depends on the state
        produced by the previous one.

        Parameters
        ----------
        step_name
            Which step to run. One of:

            - ``"qc"`` — quality control + CPM normalisation on the WT
              sample.
            - ``"nc"`` — PC network construction on the WT QC matrix.
              Writes ``self.network_dict["WT"]`` and
              ``self.shared_gene_names``.
            - ``"td"`` — tensor decomposition of the WT networks plus
              :func:`strict_direction` pruning controlled by
              ``self.strict_lambda``.
            - ``"ko"`` — produce the KO tensor according to
              ``self.ko_method``. ``kwargs`` may contain ``ko_genes``
              to override ``self.ko_genes`` for this call.
            - ``"ma"`` — manifold alignment of WT vs. KO tensors.
            - ``"dr"`` — differential regulation from the aligned
              manifold.
        **kwargs
            One-shot overrides for the step. When non-empty these
            replace the corresponding ``*_kws`` dict on the instance
            for this call only. For ``"ko"`` an explicit ``ko_genes``
            kwarg is consumed before the override logic.

        Raises
        ------
        ValueError
            If ``step_name`` is not one of the six values above, or if
            ``self.ko_method`` is unrecognised during the KO step.
        """
        start_time = time.perf_counter()
        if step_name == "qc":
            self._QC("WT", **self._step_kws("qc_kws", kwargs))
            self.QC_dict["WT"] = cpm_norm(self.QC_dict["WT"])
            print("finish QC: WT")
        elif step_name == "nc":
            self._make_networks("WT", self.QC_dict["WT"], **self._step_kws("nc_kws", kwargs))
            self.shared_gene_names = self.QC_dict["WT"].index.to_list()
        elif step_name == "td":
            self._tensor_decomp("WT", self.shared_gene_names, **self._step_kws("td_kws", kwargs))
            self.tensor_dict["WT"] = strict_direction(self.tensor_dict["WT"], self.strict_lambda).T.copy()
        elif step_name == "ko":
            self.tensor_dict["WT"] = _fill_dataframe_diagonal(self.tensor_dict["WT"], 0)
            ko_kwargs = dict(self.ko_kws)
            if kwargs.get("ko_genes") is not None:
                ko_genes = kwargs.pop("ko_genes")
            else:
                ko_genes = self.ko_genes
            ko_kwargs.update(kwargs)
            self._get_ko_tensor(ko_genes, **ko_kwargs)
        elif step_name == "ma":
            self.manifold = manifold_alignment(self.tensor_dict["WT"],
                                               self.tensor_dict["KO"],
                                               **self._step_kws("ma_kws", kwargs))
            self.step_comps["ma"] = self.manifold
        elif step_name == "dr":
            self.d_regulation = d_regulation(self.manifold, **self._step_kws("dr_kws", kwargs))
            self.step_comps["dr"] = self.d_regulation
        else:
            raise ValueError("No such step")
        print(f"process {step_name} finished in {time.perf_counter() - start_time} secs.")

    def build(self) -> pd.DataFrame:
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
