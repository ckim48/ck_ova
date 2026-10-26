"""Shared-scale half coefficients with exact algebraic residual correction."""
import numpy as np


def encode_residual(vectors, basis, public_scale):
    if not np.isfinite(public_scale) or public_scale <= 0:
        raise ValueError('A positive shared scale from earlier public states is required.')
    normalized = (vectors@basis)/public_scale
    if np.any(np.abs(normalized) > np.finfo(np.float16).max):
        raise ValueError('Half coefficient overflow; no silently dropped updates.')
    bits = normalized.astype(np.float16).view(np.uint16)
    decoded = bits.view(np.float16).astype(np.float64)*public_scale
    residual = vectors-decoded@basis.T
    return residual, bits, decoded


def reserve_symbols(original, dimensions, rank, remote_users):
    # Exactly 16 coefficient bits at the same original ideal0.5bit/symbol.
    each = 32*rank
    total = each*remote_users
    removed = np.full(len(original), total//len(original))
    removed[:total % len(original)] += 1
    data = original-removed
    if np.any(data < dimensions):
        raise ValueError('Insufficient remaining time for the analog mappings.')
    assert int(data.sum())+total == int(original.sum())
    return data, each, total


def control_resources(channel, payload_bits, control_symbols, symbol_rate):
    if payload_bits == 0:
        return np.zeros_like(channel), np.zeros_like(channel)
    target = np.expm1(np.log(2)*payload_bits/control_symbols)
    power = target/(channel*channel)
    return power, control_symbols/symbol_rate*power
