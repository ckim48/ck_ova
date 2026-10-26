"""Physical single-aggregate perturbations after exactly decoded coefficients."""
import numpy as np


def moments(blocked, design, dimension):
    users = design['users']; rho = design['rho']
    gain = np.divide(design['receiver']*design['channels']*design['analog_rms'], rho,
                     out=np.zeros_like(rho), where=rho > 0)
    bias = ((gain-1)[..., None]*blocked[users]).sum(axis=0)
    variance = design['receiver']**2/(2*design['data_symbols'])
    width = blocked.shape[-1]
    valid = np.minimum(width, np.maximum(dimension-np.arange(len(variance))*width, 0))
    analytic_rmse = float(np.sqrt(np.square(bias.reshape(-1)[:dimension]).sum()+np.sum(valid*variance)))
    return bias, variance, analytic_rmse


def perturbation(blocked, design, dimension, seed, standard_noise=None):
    bias, variance, _ = moments(blocked, design, dimension)
    noise = np.random.default_rng(seed).normal(size=bias.shape) if standard_noise is None else standard_noise
    assert noise.shape == bias.shape
    return (bias+np.sqrt(variance)[:, None]*noise).reshape(-1)[:dimension]
