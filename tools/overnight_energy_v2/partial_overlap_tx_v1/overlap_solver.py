"""Partial-overlap AirComp/NOMA; uplink transmit energy only, perfect SIC.

Analog occupies la symbols inside the digital ld-symbol frame. Digital payload
is split between the overlap and digital-only tail, with independent ideal codes.
For each prescribed payload split and receiver, the amplitude step is convex.
All reported designs undergo a separate rate, RMS, power and battery audit.
"""
import numpy as np
import cvxpy as cp
from wireless_init_v11 import receive_optimum


def received_coefficients(bits, symbols, h):
    """Exact minimum received powers per unit noise for a fixed SIC order."""
    bits = np.asarray(bits, float)
    result = np.zeros_like(bits)
    for m, length in enumerate(symbols):
        active = np.flatnonzero(bits[:, m] > 1e-12)
        if not len(active):
            continue
        if length <= 0:
            return None
        order = active[np.argsort(-h[active], kind='stable')]
        later = 0.0
        for u in order[::-1]:
            gamma = np.expm1(np.log(2)*bits[u, m]/length)
            result[u, m] = gamma*(1+later)
            later += result[u, m]
    return result


def audit(a, po, pt, rx, h, rho, bits, la, ld, eta, dims, eps, c):
    H = h[:, None]
    lt = ld-la
    pmax = c['power_limit_w']
    W = c['symbol_rate_hz']
    bound_power = (a+np.sqrt(po))**2
    energy = (la/W*bound_power+lt/W*pt).sum(axis=1)
    expected = (la/W*(a*a+po)+lt/W*pt).sum(axis=1)
    error = float(np.linalg.norm(np.abs(rx*H*a-rho).sum(axis=0)
                                +rx*np.sqrt(dims/(2*la))))
    margins = []
    for length, power, payload, analog_on in [(la, po, eta*bits, True),
                                             (lt, pt, (1-eta)*bits, False)]:
        for m in range(len(la)):
            active = np.flatnonzero(payload[:, m] > 1e-12)
            order = active[np.argsort(-h[active], kind='stable')]
            q = h*h*power[:, m]
            ia = float((h@a[:, m])**2) if analog_on else 0.0
            for j, u in enumerate(order):
                cap = length[m]*np.log2(1+q[u]/(1+ia+q[order[j+1:]].sum()))
                margins.append(float(cap-payload[u, m]))
    minimum_margin = min(margins, default=0.0)
    valid = bool(np.all(np.isfinite(energy)) and np.all(la >= dims)
                 and np.all(ld >= la) and np.all(ld/W <= c['coherence_time_s']+1e-12)
                 and np.max(bound_power) <= pmax*(1+2e-5)
                 and np.max(pt) <= pmax*(1+2e-5)
                 and np.max(energy) <= c['available_energy_j']*(1+2e-5)
                 and error <= eps*(1+2e-5) and minimum_margin >= -2e-3)
    return dict(passed=valid, error_bound=error,
                minimum_payload_margin_bits=minimum_margin,
                maximum_overlap_power_bound_w=float(np.max(bound_power)),
                maximum_tail_power_w=float(np.max(pt)),
                maximum_user_tx_energy_bound_j=float(np.max(energy)),
                tx_energy_bound_j=float(energy.sum()),
                tx_energy_expected_j=float(expected.sum()),
                analog_tx_energy_j=float((la/W*a*a).sum()),
                digital_overlap_tx_energy_j=float((la/W*po).sum()),
                digital_tail_tx_energy_j=float((lt/W*pt).sum()),
                waveform_cross_term_bound_j=float((la/W*2*a*np.sqrt(po)).sum()))


def optimize_partial(head, channels, rho, bits, dims, la, ld, eps, c,
                     split_grid=(0., .025, .05, .1, 1/7, .25), warm_receivers=()):
    assert c['sic_residual_fraction'] == 0, 'Version 1 assumes perfect SIC.'
    users = np.delete(np.arange(len(channels)), head)
    h = channels[users, head]
    if np.any(h <= 0):
        return None
    H = h[:, None]
    rho = rho[users]
    bits = bits[users]
    la = np.asarray(la, float); ld = np.asarray(ld, float); lt = ld-la
    if np.any(la < dims) or np.any(lt < 0):
        return None
    W = c['symbol_rate_hz']; P = c['power_limit_w']
    reference = np.maximum(np.max(rho/(H*np.sqrt(P)*.3), axis=0), 1e-10)
    starts = [reference*v for v in np.geomspace(.3, 6, c['receive_initializations'])]
    starts += [np.asarray(v).copy() for v in warm_receivers]
    best = None
    for eta in split_grid:
        co = received_coefficients(eta*bits, la, h)
        ct = received_coefficients((1-eta)*bits, lt, h)
        if co is None or ct is None:
            continue
        pt = ct/(H*H)
        if np.any(pt > P*(1+1e-9)) or np.any(co/(H*H) > P*(1+1e-9)):
            continue
        alpha = np.sqrt(co)/H
        tail_energy = (lt/W*pt).sum(axis=1)
        if np.any(tail_energy > c['available_energy_j']):
            continue
        for initial_rx in starts:
            rx = initial_rx.copy(); previous = float('inf')
            for iteration in range(c['max_ao_iterations']+1):
                a = cp.Variable(rho.shape, nonneg=True)
                t = cp.Variable(rho.shape, nonneg=True)
                interference_amplitude = cp.sum(cp.multiply(H, a), axis=0)
                effective_noise = cp.norm(cp.vstack([np.ones(len(la)), interference_amplitude]), axis=0)
                constraints = [a+t <= np.sqrt(P), cp.multiply(rho == 0, a) == 0,
                               cp.multiply(bits == 0, t) == 0]
                for u in range(len(users)):
                    constraints.append(cp.multiply(alpha[u], effective_noise) <= t[u])
                error = cp.sum(cp.abs(cp.multiply(rx*H, a)-rho), axis=0)+rx*np.sqrt(dims/(2*la))
                constraints.append(cp.norm(error/eps) <= 1)
                e = cp.sum(cp.multiply(la/W, cp.square(a+t)), axis=1)+tail_energy
                constraints.append(e <= c['available_energy_j'])
                problem = cp.Problem(cp.Minimize(cp.sum(e)/(ld.sum()/W)), constraints)
                try:
                    problem.solve(solver='CLARABEL', max_iter=200)
                except cp.error.SolverError:
                    break
                if problem.status not in ('optimal', 'optimal_inaccurate') or a.value is None:
                    break
                av = np.maximum(a.value, 0)
                # Recompute the exact least powers rather than trusting solver slack.
                po = co/(H*H)*(1+(H*av).sum(axis=0)**2)
                au = audit(av, po, pt, rx, h, rho, bits, la, ld, eta, dims, eps, c)
                if not au['passed']:
                    break
                value = au['tx_energy_bound_j']
                if value > previous+max(1e-9, previous*1e-5):
                    break
                if best is None or value < best['tx_energy_bound_j']:
                    best = dict(**au, head=head, users=users, analog=av,
                                digital_overlap_power=po, digital_tail_power=pt,
                                receiver=rx.copy(), payload_overlap_fraction=float(eta),
                                analog_symbols=la.copy(), digital_symbols=ld.copy())
                if np.isfinite(previous) and previous-value <= max(1e-9, previous*1e-5):
                    break
                previous = value
                rx = np.array([receive_optimum(h, av[:, m], rho[:, m],
                              np.sqrt(dims[m]/(2*la[m]))) for m in range(len(la))])
    return best
