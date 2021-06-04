import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.stats import chi2

from scTenifold.core.dim_reduction import prepare_PCA_dfs, prepare_embedding_dfs


def plot_network_graph(network: np.ndarray,
                       weight_thres=0.1,
                       con_thres=0):
    network = abs(network.copy())
    network[network < weight_thres] = 0
    valid_rows, valid_cols = (network.sum(axis=1) > con_thres), (network.sum(axis=0) > con_thres)
    network = network[valid_rows,:][:, valid_cols]
    print(network.shape)
    G = nx.from_numpy_array(network)
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, nodelist=[0], alpha=0.4)
    nx.draw_networkx_nodes(G, pos,
                           node_size=10,
                           cmap=plt.cm.Reds_r)
    plt.show()


def plot_network_heatmap(network, figsize=(12, 12)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(network, center=0.0, ax=ax)


def plot_qqplot(df,
                exp_col="FC",
                stat_col="adjusted p-value",
                plot_qqline=True,
                sig=0.1):
    the_col = "Theoretical quantiles"
    len_x = df.shape[0]
    data = df.loc[:, [exp_col, stat_col]]
    data["significant"] = data[stat_col].apply(lambda x: x < sig)
    data.sort_values(exp_col, inplace=True)
    data[the_col] = chi2.ppf(q=np.linspace(0, 1, len_x + 2)[1:-1], df=1)
    sns.scatterplot(data=data, x="Theoretical quantiles", y=exp_col, hue="significant")
    if plot_qqline:
        xl_1, xl_2 = plt.gca().get_xlim()
        x1, x2 = data[the_col].quantile(0.25), data[the_col].quantile(0.75)
        y1, y2 = data[exp_col].quantile(0.25), data[exp_col].quantile(0.75)
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        plt.plot([xl_1, xl_2],
                 [slope * xl_1 + intercept, slope * xl_2 + intercept])
        plt.xlim([xl_1, xl_2])
    plt.show()


def plot_embedding(df,
                   groups: dict,
                   method: str = "UMAP",
                   plot_2D: bool = True,
                   figsize: tuple = (8, 8),
                   title: str = None,
                   palette: str = "muted",
                   **kwargs):
    colors = sns.color_palette(palette)
    fig, ax = plt.subplots(figsize=figsize)
    if method == "PCA":
        feature_df, exp_var_df, component_df = prepare_PCA_dfs(df, **kwargs)
        emb_name = "PC"
    else:
        feature_df = prepare_embedding_dfs(df, reducer=method)
        emb_name = method

    print(feature_df.columns)
    for i, (group_name, sample_names) in enumerate(groups.items()):
        em1, em2 = np.array([feature_df.loc[name, '{} 1'.format(emb_name)] for name in sample_names]), \
                   np.array([feature_df.loc[name, '{} 2'.format(emb_name)] for name in sample_names])

        if plot_2D:
            ax.scatter(em1, em2, s=1, label=group_name, c=[colors[i]])
        else:
            em3 = np.array([feature_df.loc[name, '{} 3'.format(emb_name)] for name in sample_names])
            ax.scatter(em1, em2, em3, s=1, label=group_name, c=[colors[i]])

    x_label = '{} 1'.format(emb_name)
    y_label = '{} 2'.format(emb_name)
    z_label = None if plot_2D else '{} 3'.format(emb_name)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if z_label is not None:
        ax.set_zlabel(z_label)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()
