import pandas as pd
import numpy as np
from functools import partial
from warnings import warn


def _check_features(df, features):
    valid_features = set(df.index) & set(features)
    if len(features) != len(valid_features):
        warn(f"Found {len(features) - len(valid_features)} invalid features (e.g. not shown in the dataframe)")
    return valid_features


def calc_auc(rank_val: pd.Series, max_rank: int):
    insig_part = rank_val > max_rank
    if all(insig_part):
        return 0
    else:
        rank_val[insig_part] = max_rank + 1
        rank_sum = sum(rank_val)
        n = rank_val.shape[0]
        u_val = rank_sum - (n * (n + 1)) / 2
        auc = 1 - u_val / (n * max_rank)
        return auc


def calc_U_stat_df(features, df: pd.DataFrame, neg_features=None, max_rank=1500, w_neg=1):
    if neg_features is None:
        neg_features = []
    pos_features = list(set(features) - set(neg_features))
    if len(pos_features) > 0:
        pos = df.reindex(index=pos_features).apply(partial(calc_auc, max_rank=max_rank), axis=0).values
    else:
        pos = np.zeros(shape=(df.shape[2],))

    if len(neg_features) > 0:
        neg = df.reindex(index=neg_features).apply(partial(calc_auc, max_rank=max_rank), axis=0).values
    else:
        neg = np.zeros(shape=(df.shape[2],))
    diff = pos - w_neg * neg
    diff[diff < 0] = 0
    return diff


def calc_Uscore(df: pd.DataFrame,
                features, neg_features, max_rank=1500, w_neg=1, ties_method="average"):
    ranked_df = df.rank(ascending=False, method=ties_method)
    features = _check_features(df, features)
    cell_auc = calc_U_stat_df(features, df,
                              neg_features=neg_features,
                              max_rank=max_rank,
                              w_neg=w_neg)
    return pd.DataFrame(cell_auc, columns=ranked_df.columns)