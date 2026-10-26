"""Four physical access schemes; canonical Q8 requests and explicit failures."""
import time
import numpy as np
import cvxpy as cp
from phase_split_grid import make_inputs
from wireless_init_v11 import optimize_head, quantize, receive_optimum
from transport import recover, energy_accounting
from receiver_control import control, sensitivity_bound

FRACTIONS = (.125, .25, .375, .5, .625, .75, .875)


def oma_solve(bits, h, symbols, config, overhead, analog_rho=None,
              analog_symbols=None, receiver=None, epsilon=None):
    """Joint TDMA time/energy and optional analog amplitudes for fixed receiver.

    Channels are noise-normalized. OMA time fractions/powers and per-user
    batteries are optimized jointly, not checked only after an unconstrained fit.
    """
    bits = np.asarray(bits, float)
    n, blocks = bits.shape
    if n == 0 or np.any(np.asarray(h) <= 0):
        return None
    H = np.asarray(h)[:, None]
    symbols = np.asarray(symbols, float)
    W = config['symbol_rate_hz']
    pmax = config['power_limit_w']
    if np.any(symbols <= 0):
        return None
    t = cp.Variable((n, blocks), nonneg=True)
    q = cp.Variable((n, blocks), nonneg=True)
    constraints = [cp.sum(t, axis=0) <= 1, q <= pmax*t,
                   cp.multiply(bits == 0, t) == 0,
                   cp.multiply(bits == 0, q) == 0,
                   cp.rel_entr(t, t+cp.multiply(H**2, q))
                   + np.log(2)*bits/symbols <= 0]
    digital_energy = cp.multiply(symbols/W, q)
    energy = digital_energy
    a = None
    if analog_rho is not None:
        rho = np.asarray(analog_rho)
        analog_symbols = np.asarray(analog_symbols, float)
        a = cp.Variable(rho.shape, nonneg=True)
        constraints += [a <= np.sqrt(pmax), cp.multiply(rho == 0, a) == 0]
        energy = energy + cp.multiply(analog_symbols/W, cp.square(a))
        mismatch = cp.sum(cp.abs(cp.multiply(receiver*H, a)-rho), axis=0)
        noise = receiver*np.sqrt(config['block_dimension']/(2*analog_symbols))
        constraints += [cp.norm((mismatch+noise)/epsilon) <= 1]
    constraints += [cp.sum(energy, axis=1)+overhead <= config['available_energy_j']]
    problem = cp.Problem(cp.Minimize(cp.sum(energy)/(symbols.sum()/W)), constraints)
    try:
        problem.solve(solver='CLARABEL', max_iter=250)
    except cp.error.SolverError:
        return None
    if problem.status not in ('optimal', 'optimal_inaccurate') or t.value is None:
        return None
    tv = np.maximum(t.value, 0); qv = np.maximum(q.value, 0)
    # Independently re-evaluate Shannon capacity and all resource constraints.
    ratio = np.divide(H**2*qv, tv, out=np.zeros_like(tv), where=tv > 0)
    capacity = symbols*tv*np.log2(1+ratio)
    if np.any(capacity + 2e-4 < bits) or np.any(tv.sum(axis=0) > 1+1e-6):
        return None
    if np.any(qv > pmax*tv + 1e-7):
        return None
    ev = (symbols/W*qv).sum(axis=1)
    av = None
    bound = 0.0
    if a is not None:
        av = np.maximum(a.value, 0)
        if np.any(av**2 > pmax*(1+1e-6)):
            return None
        ev += (analog_symbols/W*av**2).sum(axis=1)
        bound = float(np.linalg.norm(np.abs(receiver*H*av-rho).sum(axis=0)
                      +receiver*np.sqrt(config['block_dimension']/(2*analog_symbols))))
        if bound > epsilon*(1+1e-5):
            return None
    if np.any(ev+overhead > config['available_energy_j']*(1+1e-6)):
        return None
    return dict(energy_by_user=ev, analog=av, fractions=tv, powers_weighted=qv,
                error_bound=bound, minimum_rate_margin=float(np.min(capacity-bits)))


def analog_oma(head, channels, rho, payload, dims, total, eps, config):
    users = np.delete(np.arange(len(channels)), head)
    overhead = np.full(len(users), config['base_overhead_j'])
    if config['base_overhead_j']+config['head_overhead_j'] > config['available_energy_j']:
        return None
    best = None
    for fraction in FRACTIONS:
        la = np.floor(total*fraction).astype(int); ld = total-la
        if np.any(la < dims) or np.any(ld <= 0):
            continue
        # Necessary OMA peak-power test, followed by joint convex allocation.
        h = channels[users, head]
        min_time = payload[users]/np.log2(1+h[:, None]**2*config['power_limit_w'])
        if np.any(min_time.sum(axis=0) > ld*(1+1e-10)):
            continue
        reference = np.maximum(np.max(rho[users]/(h[:, None]*np.sqrt(config['power_limit_w'])*.3), axis=0), 1e-10)
        for mult in np.geomspace(.3, 6, config['receive_initializations']):
            rx = reference*mult
            candidate = None
            old = float('inf')
            for iteration in range(config['max_ao_iterations']+1):
                x = oma_solve(payload[users], h, ld, config, overhead,
                              rho[users], la, rx, eps)
                if x is None:
                    break
                val = float(x['energy_by_user'].sum())
                if val > old+max(1e-10, old*1e-6):
                    break
                candidate = dict(x, receiver=rx.copy(), head=head, users=users,
                                 symbols=la, fraction=fraction, energy_j=val)
                if np.isfinite(old) and old-val <= max(1e-10, old*1e-5):
                    break
                old = val
                rx = np.asarray([receive_optimum(h, x['analog'][:, m], rho[users, m],
                                                np.sqrt(dims[m]/(2*la[m])))
                                 for m in range(len(dims))])
            if candidate is not None and (best is None or candidate['energy_j'] < best['energy_j']):
                best = candidate
    return best


def communicate(learning, unlearning, channels, config, method, seed, gradient=None):
    started = time.perf_counter()
    n, dim = learning.shape
    l, u, decoded, delta, payload, rho, dims, total = make_inputs(learning, unlearning, config)
    canonical = decoded.reshape(n, -1)[:, :dim]
    active = np.flatnonzero(np.any(unlearning != 0, axis=1))
    aggregate = learning.sum(axis=0)
    epsilon = max(config['absolute_tolerance_floor'], float(config['relative_tolerance']*np.linalg.norm(rho.sum(axis=0))))
    full_time = float(total.sum()/config['symbol_rate_hz'])
    if full_time > config['round_uplink_deadline_s']+1e-10 or np.any(total/config['symbol_rate_hz'] > config['coherence_time_s']+1e-10):
        raise ValueError('Frame violates the common deadline or coherence limit.')
    ctrl = len(active) > 0 and method != 'aircomp_only'
    target = aggregate+canonical.sum(axis=0)
    lip = 1.0
    controller_info = None
    if ctrl:
        target, controller_info = control(aggregate, canonical[active], gradient, 1.)
        lip = float(sensitivity_bound(canonical[active], gradient))
    analog_epsilon = epsilon/lip
    rng = np.random.default_rng(seed)
    overhead_total = n*config['base_overhead_j']+config['head_overhead_j']
    best = None
    diagnostics = []
    # Nonrequest rounds share the same analog design for all analog methods.
    actual_mode = method if len(active) else ('oma_only' if method == 'oma_only' else 'hybrid')
    for head in range(n):
        users = np.delete(np.arange(n), head)
        item = dict(head=head, status='no_feasible_design_found')
        candidate = None
        if actual_mode == 'oma_only':
            for bits in (4, 6, 8, 10, 12, 16, 32):
                if bits == 32:
                    ql = l.astype(np.float32).astype(np.float64)
                    dl = np.linalg.norm(ql-l, axis=-1)
                    pl = np.where(np.any(l != 0, axis=-1), dims[None, :]*32, 0)
                else:
                    ql, dl, pl = quantize(l, bits, config['scale_metadata_bits'])
                bound = float(np.linalg.norm(dl[users].sum(axis=0))*lip)
                if bound > epsilon*(1+1e-8):
                    continue
                tx = oma_solve(pl[users]+payload[users], channels[users, head], total, config,
                               np.full(len(users), config['base_overhead_j']))
                if tx is None or config['base_overhead_j']+config['head_overhead_j'] > config['available_energy_j']:
                    continue
                a_hat = (l[head]+ql[users].sum(axis=0)).reshape(-1)[:dim]
                candidate = dict(energy_j=float(tx['energy_by_user'].sum()+overhead_total),
                                 analog_hat=a_hat, error_bound=bound/lip, head=head,
                                 learning_bits=bits, energy_expected_j=float(tx['energy_by_user'].sum()+overhead_total))
                break  # More bits have no smaller ideal-Shannon energy for the same slot budget.
        elif actual_mode == 'aircomp_oma':
            d = analog_oma(head, channels, rho, payload, dims, total, analog_epsilon, config)
            if d is not None:
                received = l[head].copy()
                noise_rng = np.random.default_rng(seed)
                for m in range(len(dims)):
                    gain = d['receiver'][m]*channels[users, head]*d['analog'][:, m]/np.where(rho[users, m] > 0, rho[users, m], 1)
                    received[m] += (gain[:, None]*l[users, m]).sum(axis=0)
                    received[m] += noise_rng.normal(0, d['receiver'][m]/np.sqrt(2*d['symbols'][m]), received.shape[1])
                candidate = dict(energy_j=d['energy_j']+overhead_total, analog_hat=received.reshape(-1)[:dim],
                                 error_bound=d['error_bound'], head=head, fraction=d['fraction'],
                                 energy_expected_j=d['energy_j']+overhead_total)
        else:
            lb = l.copy(); rb = rho; pl = payload
            if actual_mode == 'aircomp_only':
                lb = l+decoded; rb = np.linalg.norm(lb, axis=-1); pl = np.zeros_like(payload)
            d = optimize_head(head, channels, rb, np.zeros_like(delta), pl, dims, total,
                              analog_epsilon, config, config['sic_residual_fraction'], 'hybrid', digital_symbols=total)
            if d is not None:
                received = recover(d, lb, np.zeros_like(lb), np.zeros_like(lb),
                                   np.random.default_rng(seed))[:dim]
                e = energy_accounting(d, config)
                candidate = dict(energy_j=d.energy_j, analog_hat=received, error_bound=d.error_bound,
                                 head=head, energy_expected_j=e['expected_energy_random_codewords_j'])
        if candidate is not None:
            item.update(status='feasible', energy_j=candidate['energy_j'])
            if best is None or candidate['energy_j'] < best['energy_j']:
                best = candidate
        diagnostics.append(item)
    if best is None:
        return None, dict(status='no_feasible_design_found', head_diagnostics=diagnostics,
                          uplink_budget_s=full_time, epsilon=epsilon,
                          optimizer_wall_s=time.perf_counter()-started,
                          required_requests=int(len(active)), individual_requests_delivered=0)
    delivered = best.pop('analog_hat')
    if ctrl:
        delivered, controller_info = control(delivered, canonical[active], gradient, 1.)
    stats = dict(status='feasible', energy_bound_j=best.pop('energy_j'),
                 expected_energy_j=best.pop('energy_expected_j'), uplink_s=full_time,
                 epsilon=epsilon, controller_lipschitz=lip, **best,
                 application_error_bound=best['error_bound']*lip,
                 realized_update_error=float(np.linalg.norm(delivered-target)),
                 head_diagnostics=diagnostics, controller=controller_info,
                 optimizer_wall_s=time.perf_counter()-started,
                 required_requests=int(len(active)), individual_requests_delivered=int(len(active)) if ctrl else 0,
                 task_eligible=bool(len(active) == 0 or ctrl))
    return delivered, stats
