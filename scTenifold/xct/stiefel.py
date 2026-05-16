import torch


def skew(M,Z):
    ''' return the skew-symmetric part of $M^T Z$'''
    return 0.5 * (M.t()@Z - Z.t()@M)


def proj_stiefel(M,Z):

    MskewMTZ = M@skew(M,Z)
    IMMTZ = (torch.eye(len(M)) - M@M.t())@Z
    return MskewMTZ + IMMTZ


def rand_stiefel(n,p):
    """
    Generate random Stiefel point using qr of random normally distributed
    matrix
    """
    X = torch.randn(n, p)
    q, r = torch.qr(X)
    return q


def retr_stiefel(Z):
    # Project onto Stiefel Manifold
    u, s, v = torch.svd(Z, some=True)
    return u@v.t()
