from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, MDS, SpectralEmbedding, LocallyLinearEmbedding
import umap
from sklearn.preprocessing import StandardScaler
import pandas as pd
from enum import Enum


class Reducer(Enum):
    TSNE = "TSNE"
    Isomap = "Isomap"
    MDS = "MDS"
    SpectralEmbedding = "SpectralEmbedding"
    LocallyLinearEmbedding = "LocallyLinearEmbedding"
    UMAP = "UMAP"


REDUCER_DICT = {Reducer.TSNE: TSNE,
                Reducer.MDS: MDS,
                Reducer.Isomap: Isomap,
                Reducer.LocallyLinearEmbedding: LocallyLinearEmbedding,
                Reducer.SpectralEmbedding: SpectralEmbedding,
                Reducer.UMAP: umap.UMAP}


def prepare_PCA_dfs(feature_df,
                    transform_func=None,
                    n_components=None,
                    standardize=True):
    if transform_func is not None:
        x = transform_func(feature_df)
    else:
        x = feature_df
    x = StandardScaler().fit_transform(x.values.T) if standardize else x.values.T
    pca = PCA(n_components=n_components)
    if not n_components:
        n_components = min(x.shape[0], x.shape[1])
    principal_components = pca.fit_transform(x)
    final_df = pd.DataFrame(data=principal_components,
                            columns=[f'PC {num + 1}' for num in range(principal_components.shape[1])],
                            index=feature_df.columns)
    exp_var_df = pd.DataFrame(data=pca.explained_variance_ratio_,
                              index=[f'PC {num + 1}' for num in range(n_components)])
    component_df = pd.DataFrame(data=pca.components_.T,
                                columns=[f'PC {num + 1}' for num in range(n_components)],
                                index=feature_df.index)
    return final_df, exp_var_df, component_df


def prepare_embedding_dfs(feature_df,
                          transform_func=None,
                          n_components=2,
                          reducer="TSNE",
                          standardize=True, **kwargs):
    if transform_func:
        x = transform_func(feature_df.values)
    else:
        x = feature_df.values
    if isinstance(reducer, str):
        reducer = Reducer(reducer)
    sample_names = feature_df.columns.to_list()
    x = StandardScaler().fit_transform(x.T) if standardize else x.values.T
    X_embedded = REDUCER_DICT[reducer](n_components=n_components, **kwargs).fit_transform(x)
    df = pd.DataFrame(X_embedded,
                      columns=["{reducer} {i}".format(reducer=reducer.value, i=i) for i in range(1, n_components + 1)],
                      index=sample_names)
    return df
