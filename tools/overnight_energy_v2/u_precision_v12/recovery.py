"""Independent phase-noise recovery of the saved two-analog design."""
import numpy as np


def moments(blocked,design):
    a,s,n,rho,h,users=[design[k] for k in ['analog_rms','receiver','symbols','rho','channels','users']]
    gain=np.divide(s[:,None,:]*h[None,:,:]*a,rho,out=np.zeros_like(rho),where=rho>0)
    bias=((gain-1)[...,None]*blocked[:,users]).sum(axis=1)
    variance=np.divide(s*s,2*n,out=np.zeros_like(s),where=n>0)
    return bias,variance


def stream_errors(blocked,design,dimension,seed,standard_noise=None):
    bias,variance=moments(blocked,design)
    z=np.random.default_rng(seed).normal(size=bias.shape) if standard_noise is None else standard_noise
    assert z.shape==bias.shape
    return (bias+np.sqrt(variance)[...,None]*z).reshape(2,-1)[:,:dimension]


def analytic_sum_rmse(blocked,design,dimension):
    bias,variance=moments(blocked,design);width=blocked.shape[-1]
    valid=np.minimum(width,np.maximum(dimension-np.arange(bias.shape[1])*width,0))
    return float(np.sqrt(np.square(bias.sum(axis=0).reshape(-1)[:dimension]).sum()+np.sum(valid*variance.sum(axis=0))))
