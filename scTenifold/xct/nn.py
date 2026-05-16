from __future__ import annotations

import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from tqdm import tqdm

from .stiefel import proj_stiefel

logger = logging.getLogger(__name__)


def set_seed(seed: int = 0) -> None:
    """Set all RNG seeds for reproducible training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class Net(nn.Module):
    """Define the neural network"""
    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        h1_sigmoid = self.linear1(x).sigmoid()
        h2_sigmoid = self.linear2(h1_sigmoid).sigmoid()
        y_pred = self.linear3(h2_sigmoid)
        return y_pred


class ManifoldAlignmentNet: # trainer
    def __init__(self,
                 data_arr,
                 w: coo_matrix,
                 n_dim,
                 layers):
        set_seed()
        self.n_models, self.data_arr, self.w = self._check_data(data_arr, w=w)
        self.model_dic = self.create_models(layers, n_dim)

    def _check_data(self, data_arr, w) -> (int, np.ndarray, coo_matrix):
        n_models = len(data_arr)
        if not all(isinstance(x_np, np.ndarray) for x_np in data_arr):
            raise TypeError('input a list of counts in numpy arrays with genes by cells')
        if not sum([x_np.shape[0] for x_np in data_arr]) == w.shape[0]:
            raise ValueError('input sequence of counts consistent with correspondence')
        return n_models, data_arr, w

    def create_models(self, layers, n_dim):
        layer_dic = {}
        if layers is None:
            a = 4
            for i in range(1, self.n_models + 1):
                n_h = scipy.stats.gmean([self.data_arr[i - 1].shape[1], n_dim]).astype(int)
                layer_dic[i] = [a * n_h, n_h, n_dim]
                # layer_dic[i] = [32, 16, n_dim]
        # elif len(layers) != 3:
        #     raise ValueError('input node numbers of three hidden layers')
        else:
            for i in range(1, self.n_models + 1):
                layer_dic[i] = layers

        model_dic = {}
        for i in range(1, self.n_models + 1):
            model_dic[f'model_{i}'] = Net(self.data_arr[i - 1].shape[1], *layer_dic[i])
            self.data_arr[i - 1] = torch.from_numpy(self.data_arr[i - 1].astype(np.float32))

        return model_dic

    def arch(self):
        for i in range(1, self.n_models + 1):
            print(f'model_{i}:\n', self.model_dic[f'model_{i}'])

    def save_model_states(self, file_dir):
        import os
        os.makedirs(file_dir, exist_ok = True)
        for i in range(1, self.n_models + 1):
            torch.save(self.model_dic[f'model_{i}'].state_dict(), f"{file_dir}/model_{i}.th")
            logger.info(f"save model to {file_dir}/model_{i}.th")

    def load_model_states(self, file_dir):
        for i in range(1, self.n_models + 1):
            self.model_dic[f'model_{i}'].load_state_dict(torch.load(f"{file_dir}/model_{i}.th"))
            self.model_dic[f'model_{i}'].eval()
            logger.info(f"load model from {file_dir}/model_{i}.th")

    def reload_embeds(self):
        y_preds = []
        for i in range(1, self.n_models + 1):
            y_preds.append(self.model_dic[f'model_{i}'](self.data_arr[i - 1]))
        outputs = torch.cat(y_preds[:], 0)
        u, _, v = torch.svd(outputs, some=True)
        proj_outputs = u @ v.t()
        self.projections = proj_outputs.detach().numpy()
        return self.projections

    def train(self,
              n_steps = 1000,
              lr = 0.01,
              verbose = False,
              **optim_kwargs):
        assert n_steps > 0
        self.losses = []
        L_np = scipy.sparse.csgraph.laplacian(self.w, normed=False).todense()
        L = torch.from_numpy(L_np.astype(np.float32))
        params = [self.model_dic[f'model_{i}'].parameters() for i in range(1, self.n_models + 1)]
        optimizer = torch.optim.Adam(itertools.chain(*params), lr=lr, **optim_kwargs)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=10, verbose=True)
        for i in range(1, self.n_models + 1):
            self.model_dic[f'model_{i}'].train()

        for t in tqdm(range(n_steps), desc = "training...", total = n_steps):
            # Forward pass: Compute predicted y by passing x to the model
            y_preds = []
            for i in range(1, self.n_models + 1):
                y_preds.append(self.model_dic[f'model_{i}'](self.data_arr[i - 1]))

            outputs = torch.cat(y_preds[:], 0)  # vertical concat
            # print('outputs', outputs.shape)

            # Project the output onto Stiefel Manifold
            u, _, v = torch.svd(outputs, some=True)
            proj_outputs = u @ v.t()

            # Compute loss
            loss = torch.trace(proj_outputs.t() @ L @ proj_outputs) /3000

            if t == 0 or t % 10 == 9:
                if verbose:
                    logger.info("%d  loss=%.6f", t + 1, loss.item())
            self.losses.append(loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            proj_outputs.retain_grad()  # et

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            # Project the (Euclidean) gradient onto the tangent space of Stiefel Manifold (to get Rimannian gradient)
            rgrad = proj_stiefel(proj_outputs, proj_outputs.grad)  # pt

            optimizer.zero_grad()

            # Backpropogate the Rimannian gradient w.r.t proj_outputs
            proj_outputs.backward(rgrad)  # backprop(pt)
            optimizer.step()
            # scheduler.step()

        self.projections = proj_outputs.detach().numpy()
        return self.projections

    def plot_losses(self, file_name=None):
        '''plot loss every 100 steps'''
        plt.figure(figsize=(6, 5), dpi=80)
        plt.plot(np.arange(len(self.losses)), self.losses)
        if file_name is not None:
            plt.savefig(file_name, dpi=80)
        plt.show()

    @staticmethod
    def _pair_distance(
                    projections,
                    gene_names_x,
                    gene_names_y,
                    dist_metric='euclidean'):
        '''distances of each directional pair in latent space'''
        X = projections[:len(projections) // 2, :]
        Y = projections[len(projections) // 2:, :]
        dist = scipy.spatial.distance.cdist(X, Y, metric=dist_metric)
        dist_df = pd.DataFrame(dist, index=gene_names_x, columns=gene_names_y)

        return dist_df

    def nn_aligned_dist(self,
                        projections,
                        gene_names_x,
                        gene_names_y,
                        w12_shape,
                        dist_metric='euclidean',
                        rank=False,
                        verbose: bool = True):
        '''output info of each pair'''
        if verbose:
            logger.info(f"computing pair-wise {dist_metric} distances...")
        dist_df = ManifoldAlignmentNet._pair_distance(projections, gene_names_x, gene_names_y, dist_metric=dist_metric)
        dist_df = pd.DataFrame(dist_df.stack())  # multi_index, colname 0 for dist
        dist_df = dist_df.rename_axis([1, 2]).reset_index(level=[1, 2])
        dist_df.columns = ['ligand', 'receptor', 'dist']
        dist_df.index = dist_df['ligand'] + '_' + dist_df['receptor']

        # print('adding column \'correspondence\'...')
        w12 = self.w.toarray()[:w12_shape[0], w12_shape[1]:]
        dist_df['correspondence'] = w12.reshape((w12.size,))
        del w12

        if rank:
            # print('adding column \'rank\'...')
            dist_df = dist_df.sort_values(by=['dist'])
            dist_df['rank'] = np.arange(len(dist_df)) + 1
        return dist_df

    def filtered_nn_aligned_dist(self, df_nn, candidates):
        '''filter and rank only L-R pairs'''
        if 'diff2' in df_nn.columns:
            df_nn_filtered = df_nn.loc[candidates].sort_values(by=['diff2'])  # dist difference^2 ranked L-R candidates
        else:
            df_nn_filtered = df_nn.loc[candidates].sort_values(by=['dist'])  # dist ranked L-R candidates
        logger.info("manifold aligned # of L-R pairs: %d", len(df_nn_filtered))
        df_nn_filtered['rank_filtered'] = np.arange(len(df_nn_filtered)) + 1

        return df_nn_filtered

