"""Recover physical error after separately paid exact coefficient delivery."""
import numpy as np


def moments(analog, digital, decoded, design, dimension):
    assert float(design['chi']) == 0., 'Recorded physical pilot requires zero SIC residual.'
    users = design['users']; rho = design['rho']
    gain = np.divide(design['receiver']*design['normalized_channels']*design['analog_rms'], rho,
                     out=np.zeros_like(rho), where=rho > 0)
    bias = (((gain-1)[..., None]*analog[users]).sum(axis=0)+(decoded[users]-digital[users]).sum(axis=0))
    variance = design['receiver']**2/(2*design['symbols'])
    width = analog.shape[-1]
    valid = np.minimum(width, np.maximum(dimension-np.arange(len(variance))*width, 0))
    rms = float(np.sqrt(np.square(bias.reshape(-1)[:dimension]).sum()+np.sum(valid*variance)))
    return bias, variance, rms


def perturbation(analog, digital, decoded, design, dimension, seed, noise=None):
    bias, variance, _ = moments(analog, digital, decoded, design, dimension)
    standard = np.random.default_rng(seed).normal(size=bias.shape) if noise is None else noise
    assert standard.shape == bias.shape
    return (bias+np.sqrt(variance)[:, None]*standard).reshape(-1)[:dimension]
