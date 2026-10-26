"""Locally known quantization-error norms, with paid reliable statistics."""
import numpy as np


def report_bounds(digital, decoded):
    exact = np.linalg.norm(decoded-digital, axis=-1)
    rounded = exact.astype(np.float32)
    rounded = np.where(rounded.astype(float) < exact,
                       np.nextafter(rounded, np.float32(np.inf)), rounded).astype(np.float32)
    upper = rounded.astype(float)
    assert np.all(upper >= exact)
    return upper


def reserve(channels, digital, total_symbols, head, rate):
    users = np.delete(np.arange(len(digital)), head)
    reporters = np.any(digital[users] != 0, axis=(1, 2))
    bits_per_reporter = 32*len(total_symbols)
    symbols_per_reporter = 2*bits_per_reporter  #0.5bits/complexsymbol, same chosen control modem as history pilots.
    total_control = int(reporters.sum())*symbols_per_reporter
    cut = np.full(len(total_symbols), total_control//len(total_symbols), dtype=int)
    cut[:total_control % len(total_symbols)] += 1
    main = total_symbols-cut
    pc = np.where(reporters, np.expm1(np.log(2)*.5)/channels[users, head]**2, 0.)
    ec = pc*symbols_per_reporter/rate
    assert int(main.sum())+total_control == int(total_symbols.sum())
    return main, pc, ec, reporters, symbols_per_reporter
