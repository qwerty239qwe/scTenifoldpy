import pandas as pd


def cpm_norm(X):
    return X / X.sum(axis=0)