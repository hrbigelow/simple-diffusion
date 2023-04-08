import torch as t
import numpy as np

def _isonormal(x, mu, sigma, combomode, logmode):
    if combomode:
        new_shape = (*x.shape[:-1], *((1,) * (mu.ndim - 1)), x.shape[-1])
        x = x.reshape(*new_shape)

    scaled_diff = (x - mu) / sigma
    expon = -0.5 * (scaled_diff ** 2).sum(dim=-1)

    D = x.shape[-1]
    if logmode:
        logdenomsq = D * np.log(2 * np.pi) + t.log(sigma.prod(dim=-1))
        lognorm = - 0.5 * logdenomsq
        return lognorm + expon
    else:
        denomsq = (2 * np.pi) ** D * sigma.prod(dim=-1)
        norm = denomsq ** -0.5
        return norm * t.exp(expon)


def isonormal_pdf(x, mu, sigma):
    """
    Compute isotropic normal pdf, with one-to-one batching across x, mu, sigma
    x, mu, sigma: bdims, D
    returns: bdims
    """
    return _isonormal(x, mu, sigma, combomode=False, logmode=False)

def isonormal_logpdf(x, mu, sigma):
    """
    Compute isotropic normal logpdf, with one-to-one batching across x, mu, sigma
    x, mu, sigma: bdims, D
    returns: bdims
    """
    return _isonormal(x, mu, sigma, combomode=False, logmode=True)

def combo_isonormal_pdf(x, mu, sigma):
    """
    density of B points against P isotropic gaussian distributions of dimension D
    x: bdims, D
    mu, sigma: pdims, D
    return: bdims, pdims 
    """
    return _isonormal(x, mu, sigma, combomode=True, logmode=False)

def combo_isonormal_logpdf(x, mu, sigma):
    """
    log density of B points against P isotropic gaussian distributions of dimension D
    x: bdims, D
    mu, sigma: pdims, D
    return: bdims, pdims 
    """
    return _isonormal(x, mu, sigma, combomode=True, logmode=True)


