from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse

from .core import scTenifoldXct
from .nn import ManifoldAlignmentNet
from .stat import chi2_diff_test

logger = logging.getLogger(__name__)


class merge_scTenifoldXct:
    def __init__(self,
                *Xcts: scTenifoldXct,
                verbose: bool = True):
        self.Xcts = Xcts
        self._merge_candidates = list(set(self.Xcts[0].candidates).union(set(self.Xcts[1].candidates)))
        self.verbose = verbose
        self.n_dim = 3
        self.mu = 0.9
        # cal big W
        if self.verbose:
            logger.info("merge samples and build correspondence")
        self._W, self.W12_shape = self._build_W()

        self._nn_trainer = ManifoldAlignmentNet(self._get_data_arrs(),
                                                w=self._W,
                                                n_dim=self.n_dim,
                                                layers=None)
        if self.verbose:
            logger.info("merge_scTenifoldXct init completed")

    def _get_data_arrs(self):
        '''return a list of counts in numpy array, gene by cell'''
        return self.Xcts[0].get_data_arr() + self.Xcts[1].get_data_arr()

    def _build_W(self):
        '''build a cross-object corresponding matrix for further differential analysis'''
        w_A, w_B = self.Xcts[0].w, self.Xcts[1].w
        W12 = np.zeros((w_A.shape[0], w_B.shape[1]), float)
        scaled_diag = self.mu * (w_A.sum() + w_B.sum()) / (4 * len(W12))
        np.fill_diagonal(W12, scaled_diag)
        W = sparse.vstack([sparse.hstack([w_A, W12]),
                           sparse.hstack([W12.T, w_B])])
        return W, W12.shape

    @property
    def trainer(self):
        return self._nn_trainer

    @property
    def W(self):
        return self._W

    @property
    def merge_candidates(self):
        return self._merge_candidates

    # @profile(precision=4)
    def get_embeds(self,
                train: bool = True,
                n_steps: int = 3000,
                lr: float = 0.01,
                verbose: bool = False,
                plot_losses: bool = False,
                losses_file_name: str | None = None,
                **optim_kwargs
                ) -> np.ndarray:
        if train:
            merge_projections = self._nn_trainer.train(n_steps=n_steps, lr=lr, verbose=verbose, **optim_kwargs)
        else:
            merge_projections = self._nn_trainer.reload_embeds()
        if plot_losses:
            self._nn_trainer.plot_losses(losses_file_name)

        return merge_projections

    def plot_losses(self, **kwargs):
        self._nn_trainer.plot_losses(**kwargs)


    def nn_aligned_diff(self,
                    merge_projections: np.ndarray,
                    dist_metric: str = "euclidean",
                    rank: bool = False
                    ) -> None:
        '''pair-wise difference of aligned distance'''
        projections_split = np.array_split(merge_projections, 2)
        xct_A, xct_B = self.Xcts[0], self.Xcts[1]
        self._aligned_result_A = xct_A.trainer.nn_aligned_dist(projections_split[0],
                                                               gene_names_x=xct_A.genes[xct_A.cell_names[0]],
                                                               gene_names_y=xct_A.genes[xct_A.cell_names[1]],
                                                               w12_shape=xct_A.w12_shape,
                                                               dist_metric=dist_metric,
                                                               rank=rank,
                                                               verbose=self.verbose)
        self._aligned_result_B = xct_B.trainer.nn_aligned_dist(projections_split[1],
                                                               gene_names_x=xct_B.genes[xct_B.cell_names[0]],
                                                               gene_names_y=xct_B.genes[xct_B.cell_names[1]],
                                                               w12_shape=xct_B.w12_shape,
                                                               dist_metric=dist_metric,
                                                               rank=rank,
                                                               verbose=self.verbose)
        df_nn_all = pd.concat([self._aligned_result_A, self._aligned_result_B.drop(['ligand', 'receptor'], axis=1)], axis=1)
        # print('adding column \'diff2\'...')
        df_nn_all['diff2'] = np.square(df_nn_all['dist'].iloc[:, 0] - df_nn_all['dist'].iloc[:, 1]) #there are two 'dist' cols
        if rank:
            # print('adding column \'diff2_rank\'...')
            df_nn_all = df_nn_all.sort_values(by=['diff2'], ascending=False)
            df_nn_all['diff2_rank'] = np.arange(len(df_nn_all)) + 1
        self._aligned_diff_result = df_nn_all
        if self.verbose:
            logger.info("merged pair-wise distances")
        # return df_nn_all

    def chi2_diff_test(self,
                  dof: int = 1,
                  pval: float = 0.05,
                  cal_FDR: bool = True,
                  plot_result: bool = False,
                  ) -> pd.DataFrame:
        return chi2_diff_test(df_nn=self._aligned_diff_result,
                        df=dof,
                        pval=pval,
                        FDR=cal_FDR,
                        candidates=self._merge_candidates,
                        plot=plot_result)

    @property
    def aligned_diff_result(self):
        if self._aligned_diff_result is None:
            raise AttributeError("No aligned_diff_result created yet. "
                                 "Please call train_nn() to train the neural network to get embeddings first.")
        return self._aligned_diff_result


def main(args):
    from time import time

    import scanpy as sc

    if args.eva:
        adata = sc.datasets.pbmc3k()
        adata_WT = adata[
            np.random.choice(adata.shape[0], args.n_sample, replace=False),
            np.random.choice(adata.shape[1], args.n_feature, replace=False)].copy()
        adata_WT.obs["ident"] = ["cell_A"] * (len(adata_WT)//2) + ["cell_B"] * (args.n_sample-len(adata_WT)//2)
        adata_KO = adata_WT.copy()
        adata_KO.obs["ident"] = np.random.permutation(adata_WT.obs["ident"])
        for adata in [adata_WT, adata_KO]:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            adata.layers["log1p"] = adata.X
        xct_WT = scTenifoldXct(data = adata_WT,
                                source_celltype = 'cell_A',
                                target_celltype = 'cell_B',
                                obs_label = 'ident',
                                rebuild_GRN = args.rebuild, # timer
                                GRN_file_dir = './Net_example_dev_WT',
                                verbose = True,
                                n_cpus = args.n_cpus,
                                )
        xct_KO = scTenifoldXct(data = adata_KO,
                            source_celltype = 'cell_A',
                            target_celltype = 'cell_B',
                            obs_label = 'ident',
                            rebuild_GRN = args.rebuild, # timer
                            GRN_file_dir = './Net_example_dev_KO',
                            verbose = True,
                            n_cpus = args.n_cpus,
                            )
    else:
        from .dataLoader import build_adata
        adata = build_adata(counts_path = args.file)
        print(adata)
        ada_WT = adata[adata.obs[args.cond_label] == args.cond_WT, :].copy()
        ada_KO = adata[adata.obs[args.cond_label] == args.cond_KO, :].copy()
        del adata
        xct_WT = scTenifoldXct(data = ada_WT,
                            source_celltype = args.sender,
                            target_celltype = args.receiver,
                            obs_label = args.label,
                            rebuild_GRN = args.rebuild, # timer
                            GRN_file_dir = args.workdir,
                            verbose = args.verbose,
                            n_cpus = args.n_cpus)
        xct_KO = scTenifoldXct(data = ada_KO,
                            source_celltype = args.sender,
                            target_celltype = args.receiver,
                            obs_label = args.label,
                            rebuild_GRN = args.rebuild, # timer
                            GRN_file_dir = args.workdir,
                            verbose = args.verbose,
                            n_cpus = args.n_cpus)
    XCTs = merge_scTenifoldXct(xct_KO, xct_WT)
    start_t = time()
    emb = XCTs.get_embeds(train = True)
    print(f'training time: {time()-start_t:.2f} s')
    XCTs.nn_aligned_diff(emb)
    xcts_pairs_diff = XCTs.chi2_diff_test()
    xcts_pairs_diff.to_csv(f'{args.workdir}/{args.output}.csv')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('file', type = str)
    parser.add_argument('cond_label', type = str)
    parser.add_argument('cond_WT', type = str)
    parser.add_argument('cond_KO', type = str)
    parser.add_argument('-w', '--workdir', type = str, default = 'xct_results')
    parser.add_argument('-o', '--output', type = str, default = 'xct_enriched_diff')
    parser.add_argument('-s', '--sender', type = str, default = 'cell_A')
    parser.add_argument('-r', '--receiver', type = str, default = 'cell_B')
    parser.add_argument('-l', '--label', type = str, default = 'ident')
    parser.add_argument('--n_cpus', type = int, default = -1)
    parser.add_argument('-v', '--verbose', action = 'store_true')

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--rebuild', dest = 'rebuild', action = 'store_true')
    feature_parser.add_argument('--no-rebuild', dest = 'rebuild', action ='store_false')
    parser.set_defaults(rebuild = True)

    parser.add_argument('--eva', action = 'store_true')
    parser.add_argument('--n_sample', type = int, default = 100)
    parser.add_argument('--n_feature', type = int, default = 3000)

    args = parser.parse_args()
    main(args)
    # Example usage: see README / docs, or `sctenifoldxct-merge --help`
