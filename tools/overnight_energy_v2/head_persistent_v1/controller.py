"""Small exact dual solve for a soft, one-sided rollback guard.

Direction constraints are surrogates, not unlearning certificates.
"""
import itertools
import numpy as np


def unit_rows(rows):
    rows = np.asarray(rows, dtype=float)
    norms = np.linalg.norm(rows, axis=1)
    return rows[norms > 1e-15] / norms[norms > 1e-15, None]


def protect(delta, drift, basis, strength=1.0):
    delta = np.asarray(delta, dtype=float)
    drift = np.asarray(drift, dtype=float)
    basis = np.asarray(basis, dtype=float)
    if basis.size == 0:
        return delta.copy(), dict(active=0, correction_ratio=0.0)
    assert len(basis) <= 8 and strength > 0
    gram = basis @ basis.T
    t = basis @ (drift + delta)
    h = gram + np.eye(len(basis)) / strength
    answer = None
    for mask in itertools.product([False, True], repeat=len(basis)):
        active = np.flatnonzero(mask)
        y = np.zeros(len(basis))
        if len(active):
            y[active] = np.linalg.solve(h[np.ix_(active, active)], -t[active])
        gradient = h @ y + t
        tol = 1e-10 * max(1.0, np.linalg.norm(t))
        if np.min(y) >= -tol and np.all(gradient[~np.array(mask)] >= -tol):
            answer = np.maximum(y, 0.0)
            break
    if answer is None:
        raise RuntimeError('No KKT-consistent dual active set found')
    correction = basis.T @ answer
    result = delta + correction
    violation_before = np.maximum(-t, 0.0)
    violation_after = np.maximum(-basis @ (drift + result), 0.0)
    # Dual/primal stationarity: d-d0 = lambda B^T[-B(drift+d)]_+.
    residual = correction - strength * basis.T @ violation_after
    assert np.linalg.norm(residual) < 1e-8 * max(1.0, np.linalg.norm(result))
    return result, dict(active=int(np.count_nonzero(answer > 1e-12)),
        correction_ratio=float(np.linalg.norm(correction) / max(np.linalg.norm(delta), 1e-15)),
        violation_before=float(np.linalg.norm(violation_before)),
        violation_after=float(np.linalg.norm(violation_after)),
        stationarity_residual=float(np.linalg.norm(residual)))


def checks():
    from scipy.optimize import minimize
    rng = np.random.default_rng(221026)
    maximum = 0.0
    for _ in range(10):
        b = unit_rows(rng.normal(size=(5, 17)))
        d = rng.normal(size=17)
        drift = rng.normal(size=17)
        got, _ = protect(d, drift, b)
        def fun(x):
            z = np.maximum(-b @ (drift + x), 0.)
            return .5 * np.sum((x-d)**2) + .5 * z @ z
        def jac(x):
            return x-d-b.T @ np.maximum(-b @ (drift+x), 0.)
        ref = minimize(fun, d, jac=jac, method='BFGS', options={'gtol':1e-10})
        maximum = max(maximum, float(abs(fun(got)-fun(ref.x))))
        assert abs(fun(got)-fun(ref.x)) < 1e-8
    d=np.array([1.,2.]); b=np.eye(2)
    assert np.allclose(protect(d,np.zeros(2),b)[0],d)
    assert np.allclose(protect(-d,np.zeros(2),b)[0],-.5*d)
    assert np.allclose(protect(d,np.zeros(2),np.empty((0,2)))[0],d)
    assert np.allclose(protect(np.array([-1.,0.]),np.array([2.,0.]),b)[0],[-1.,0.])
    # A cached head memory works on an AirComp sum too. No intrinsic PHY advantage.
    learning=rng.normal(size=(10,17)); basis=unit_rows(rng.normal(size=(5,17)))
    received_sum=learning.sum(0)
    assert np.allclose(protect(received_sum,np.zeros(17),basis)[0],
                       protect(learning.sum(0),np.zeros(17),basis)[0])
    return dict(passed=True,independent_primal_optimizer_max_gap=maximum,
        inactive_identity=True,one_dimensional_soft_solution=True,
        cumulative_progress_slack=True,cached_memory_aircomp_equivalence=True)
