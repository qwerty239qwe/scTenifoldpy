import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def chi2_test(df_nn: pd.DataFrame,
            df: int = 1,
            pval: float = 0.05,
            FDR: bool = True,
            candidates: list = None,
            plot: bool = False):
    '''chi-sqaure left tail test to have enriched pairs'''
    if 'dist' not in df_nn.columns:
        raise IndexError('require resulted dataframe with column \'dist\'')
    else:
        dist2 = np.square(np.asarray(df_nn['dist']))
        dist_mean = np.mean(dist2)
        FC = df * dist2 / dist_mean
        p = scipy.stats.chi2.cdf(FC, df = df) # left tail CDF
        df_enriched = df_nn.copy()
        df_enriched['FC'] = FC
        df_enriched['p_val'] = p
        if plot:
            fc_null = df_enriched[df_enriched['correspondence'] != 0]['FC']
            df_test = df_enriched[df_enriched.index.isin(candidates)]

        if FDR:
            rej, q, _, _ = multipletests(pvals = p, alpha = pval, method = 'fdr_bh')
            df_enriched['q_val'] = q
            df_enriched = df_enriched[df_enriched['q_val'] < pval]
        else:
            df_enriched = df_enriched[df_enriched['p_val'] < pval]
        if candidates is not None:
            df_enriched = df_enriched[df_enriched.index.isin(candidates)].sort_values(by=['dist'])
            df_enriched['enriched_rank'] = np.arange(len(df_enriched)) + 1
        logger.info('Total enriched: %d / %d', len(df_enriched), len(df_nn))

        if plot:
            plt.hist(fc_null, bins=1000, color='royalblue')
            enriched_bool = df_test.index.isin(df_enriched.index)
            for fc, b in zip(df_test['FC'], enriched_bool):
                if b:
                    c = 'red'
                else:
                    c = 'gray'
                plt.axvline(fc, ls=(0, (1, 1)), linewidth=0.5, alpha=0.8, c=c)
            plt.xlabel('FC')
            plt.show()

        return df_enriched


def chi2_diff_test(df_nn: pd.DataFrame,
                df: int = 1,
                pval: float = 0.05,
                FDR: bool = True,
                candidates: list = None,
                plot: bool = False):
    '''chi-sqaure right tail test to have pairs with significant distance difference'''
    if 'diff2' not in df_nn.columns:
        raise IndexError('require resulted dataframe with column \'diff2\'')

    else:
        # dist2 = np.square(np.asarray(df_nn['diff']))
        dist_mean = np.mean(df_nn['diff2'])
        FC = df * np.asarray(df_nn['diff2']) / dist_mean
        p = 1- scipy.stats.chi2.cdf(FC, df=df)  # 1- left tail CDF
        df_enriched = df_nn.copy()
        df_enriched['FC'] = FC
        df_enriched['p_val'] = p

        if plot:
            df_test = df_enriched[df_enriched.index.isin(candidates)]

        if FDR:
            rej, q, _, _ = multipletests(pvals=p, alpha=pval, method='fdr_bh')
            df_enriched['q_val'] = q
            df_enriched = df_enriched[df_enriched['q_val'] < pval]
        else:
            df_enriched = df_enriched[df_enriched['p_val'] < pval]

        if candidates is not None:
            df_enriched = df_enriched[df_enriched.index.isin(candidates)].sort_values(by=['diff2'], ascending=False)
            df_enriched['enriched_rank'] = np.arange(len(df_enriched)) + 1
            # df_enriched['dir'] = (df_enriched['dist'].iloc[:, 1] > df_enriched['dist'].iloc[:, 0]).astype(int)
            # df_enriched['dir_sign'] = df_enriched['dir'].replace(
            #     {1: u'\u2191', 0: u'\u2193'})  # obj1 (base) vs obj2, up: pairs interaction strengthed in base
        logger.info('Total enriched: %d / %d', len(df_enriched), len(df_nn))

        if plot:
            plt.hist(df_nn['diff2'], bins=1000, color='royalblue')
            enriched_bool = df_test.index.isin(df_enriched.index)
            for fc, b in zip(df_test['diff2'], enriched_bool):
                if b:
                    c = 'red'
                else:
                    c = 'gray'
                plt.axvline(fc, ls=(0, (1, 1)), linewidth=0.5, alpha=0.8, c=c)
            plt.xlabel('diff FC')
            plt.show()

        return df_enriched


def null_test(df_nn: pd.DataFrame,
            candidates,
            filter_zeros=True,
            pval=0.05,
            plot=False):
    '''nonparametric left tail test to have enriched pairs'''
    if 'dist' not in df_nn.columns or 'correspondence' not in df_nn.columns:
        raise IndexError('require resulted dataframe with column \'dist\' and \'correspondence\'')

    else:
        dist_test = df_nn[df_nn.index.isin(candidates)].copy()
        # filter pairs with correspondence_score zero
        if filter_zeros:
            mask = df_nn['correspondence'] != 0
        else:
            mask = np.ones(len(df_nn), dtype=bool)
        dist_null = df_nn[(~df_nn.index.isin(candidates)) & (mask)]
        dist_test['p_val'] = dist_test['dist'].apply(
            lambda x: scipy.stats.percentileofscore(dist_null['dist'], x) / 100)
        df_enriched = dist_test[dist_test['p_val'] < pval].sort_values(by=['dist'])
        logger.info('Total enriched: %d / %d', len(df_enriched), len(df_nn))
        df_enriched['enriched_rank'] = np.arange(len(df_enriched)) + 1

        if plot:
            cut = np.percentile(dist_null['dist'].values, pval)  # left tail
            plt.hist(dist_null['dist'], bins=1000, color='royalblue')
            for d in dist_test['dist']:
                if d < cut:
                    c = 'red'
                else:
                    c = 'gray'
                plt.axvline(d, ls=(0, (1, 1)), linewidth=0.5, alpha=0.8, c=c)
            plt.xlabel('distance')
            plt.show()
        del dist_test
    return df_enriched
