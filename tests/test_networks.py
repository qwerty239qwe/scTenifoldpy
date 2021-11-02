import pytest
from functools import partial

import numpy as np
import scipy.stats as stats

from scTenifold.core._networks import cal_pc_coefs
from scTenifold.core._utils import get_test_df


def test_cal_pc_coefs_stability():
    df = get_test_df(n_genes=100).values
    df = np.log2(df + 1)
    df = df.T  # cells x genes
    df = (df - df.mean(axis=0)) / df.std(axis=0)

    p_scipy = partial(cal_pc_coefs, method="scipy", X=df, n_comp=3, random_state=42)
    bs = [p_scipy(i) for i in range(df.shape[1])]
    p_ = partial(cal_pc_coefs, method="sklearn", X=df, n_comp=3, random_state=42)
    bs_2 = [p_(i) for i in range(df.shape[1])]

    assert all([stats.pearsonr(b1.flatten(), b2.flatten())[0] > 0.99 for b1, b2 in zip(bs, bs_2)])