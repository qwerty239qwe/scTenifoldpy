import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import networkx as nx


def plot_network_graph(network):
    network = network[network.s]
    G = nx.from_numpy_array(network)
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, nodelist=[0], alpha=0.4)
    nx.draw_networkx_nodes(G, pos,
                           node_size=10,
                           cmap=plt.cm.Reds_r)
    plt.show()


def plot_network_heatmap(network):
    sns.heatmap(network, center=0.0)


def qqplot(df,
           exp_col="FC",
           stat_col="adjusted p-value",
           plot_qqline=True,
           sig = 0.1):
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
        slope = (y2 - y1) / (x2-x1)
        intercept = y1 - slope * x1
        plt.plot([xl_1, xl_2],
                 [slope * xl_1 + intercept, slope * xl_2 + intercept])
        plt.xlim([xl_1, xl_2])
    plt.show()