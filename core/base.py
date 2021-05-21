import numpy as np
from core.QC import scQC
from core.networks import *
from core.normalization import cpm_norm
from core.decomposition import tensor_decomp


class scBase:
    def __init__(self,
                 qc_kws=None,
                 nc_kws=None,
                 td_kws=None,
                 ma_kws=None
                 ):
        self.data_dict = {}
        self.QC_dict = {}
        self.network_dict = {}
        self.tensor_dict = {}
        self.manifold = None
        self.d_regulation = None
        self.qc_kws = {} if qc_kws is None else qc_kws
        self.nc_kws = {} if nc_kws is None else nc_kws
        self.td_kws = {} if td_kws is None else td_kws
        self.ma_kws = {} if ma_kws is None else ma_kws

    def _QC(self, label):
        self.QC_dict[label] = scQC(self.data_dict[label], **self.qc_kws)

    def _make_networks(self, label, data):
        self.network_dict[label] = make_networks(data, **self.nc_kws)

    def _tensor_decomp(self, label, gene_names):
        self.tensor_dict[label] = tensor_decomp(self.network_dict[label], gene_names, **self.td_kws)


class scTenifoldNet(scBase):
    def __init__(self, X, Y,
                 x_label, y_label,
                 qc_kws=None,
                 nc_kws=None,
                 td_kws=None,
                 ma_kws=None):
        super().__init__(qc_kws=qc_kws, nc_kws=nc_kws, td_kws=td_kws, ma_kws=ma_kws)
        self.x_label, self.y_label = x_label, y_label
        self.data_dict[x_label] = X
        self.data_dict[y_label] = Y

    def _norm(self, label):
        self.QC_dict[label] = cpm_norm(self.QC_dict[label])

    def build(self):
        print("performing QC and normalization")
        for label in self.data_dict:
            self._QC(label)
            self._norm(label)
            print("finish QC:", label)

        x_gene_names, y_gene_names = set(self.data_dict[self.x_label].index), set(self.data_dict[self.y_label].index)
        shared_gene_names = list(x_gene_names & y_gene_names)

        for label, qc_data in self.QC_dict.items():
            self._make_networks(label, data=qc_data.loc[shared_gene_names, :])
            self._tensor_decomp(label, shared_gene_names)
        tensorX = (self.tensor_dict[self.x_label] + self.tensor_dict[self.x_label].T) / 2
        tensorY = (self.tensor_dict[self.y_label] + self.tensor_dict[self.y_label].T) / 2
        self.manifold = manifold_alignment(tensorX, tensorY, **self.ma_kws)
        self.d_regulation = d_regulation(self.manifold)
        return self.d_regulation


class scTenifoldKnk(scBase):
    def __init__(self, data,
                 strict_lambda=0,
                 ko_genes=None,
                 qc_kws=None,
                 nc_kws=None,
                 td_kws=None,
                 ma_kws=None):
        super().__init__(qc_kws=qc_kws, nc_kws=nc_kws, td_kws=td_kws, ma_kws=ma_kws)
        self.data_dict["WT"] = data
        self.strict_lambda = strict_lambda
        self.ko_genes = ko_genes if ko_genes is not None else []

    def build(self):
        self._QC("WT")
        if self.QC_dict["WT"].shape[1] > 500:
            self.QC_dict["WT"] = self.QC_dict["WT"].loc[self.QC_dict["WT"].mean(axis=1) >= 0.05, :]
        else:
            self.QC_dict["WT"] = self.QC_dict["WT"].loc[self.QC_dict["WT"].sum(axis=1) >= 25, :]
        self._make_networks("WT", self.QC_dict["WT"])
        self._tensor_decomp("WT", self.QC_dict["WT"].index.to_list())
        self.tensor_dict["WT"] = strict_direction(self.tensor_dict["WT"], self.strict_lambda).T
        np.fill_diagonal(self.tensor_dict["WT"], 0)
        self.tensor_dict["KO"] = self.tensor_dict["WT"].copy()
        self.tensor_dict["KO"].loc[self.ko_genes, :] = 0
        self.manifold = manifold_alignment(self.tensor_dict["WT"], self.tensor_dict["KO"], **self.ma_kws)
        self.d_regulation = d_regulation(self.manifold)
        return self.d_regulation
