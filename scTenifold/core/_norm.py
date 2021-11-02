

def cpm_norm(X):
    return X * 1e6 / X.sum(axis=0)