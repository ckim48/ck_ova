"""Nearest update with nonnegative alignment to every observable objective update.

This is a zero-margin first-order safeguard, not a deletion certificate. It can
stall at a Pareto conflict. No clipping/orthogonalization is imposed on baselines.
"""
import numpy as np


def project(nominal, directions):
    import cvxpy as cp
    nominal = np.asarray(nominal, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, len(nominal))
    norms = np.linalg.norm(directions, axis=1)
    active = norms > 1e-14
    a = directions[active] / norms[active, None]
    scale = float(np.linalg.norm(nominal))
    diag = dict(constraints=int(active.sum()), nominal_norm=scale,
                nominal_conflicts=int(np.sum(a @ nominal < -1e-10)) if len(a) else 0)
    if scale < 1e-14 or not len(a):
        return nominal.copy(), dict(diag, status='identity', correction_ratio=0., norm_ratio=1., minimum_alignment=0.)
    x0 = nominal / scale
    before = a @ x0
    if np.min(before) >= -1e-12:
        return nominal.copy(), dict(diag, status='identity', correction_ratio=0., norm_ratio=1., minimum_alignment=float(np.min(before)))
    _, singular, vt = np.linalg.svd(a, full_matrices=False)
    rank = int(np.sum(singular > max(singular[0]*1e-10, 1e-14)))
    basis = vt[:rank]
    y0 = basis @ x0
    # Wireless noise can have a component outside the constraint span: retain it.
    perp = x0 - basis.T @ y0
    y = cp.Variable(rank)
    problem = cp.Problem(cp.Minimize(.5*cp.sum_squares(y-y0)), [a @ basis.T @ y >= 0])
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_feas=1e-10,
                  tol_gap_rel=1e-10, max_iter=300)
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        raise RuntimeError('Projection solve failed: '+problem.status)
    result = scale * (basis.T @ y.value + perp)
    alignment = float(np.min(a @ (result/scale)))
    ratio = float(np.linalg.norm(result)/scale)
    if alignment < -2e-6 or ratio > 1+2e-6:
        raise RuntimeError('Projection numerical audit failed')
    return result, dict(diag, status=problem.status, rank=rank,
        correction_ratio=float(np.linalg.norm(result-nominal)/scale),
        norm_ratio=ratio, minimum_alignment=alignment)


def global_project(learning_sum, requests, aggregate=False):
    requests = np.asarray(requests)
    active = requests[np.linalg.norm(requests, axis=1)>1e-14]
    nominal = learning_sum + requests.sum(0)
    if not len(active):
        return nominal, dict(status='no_requests', correction_ratio=0., nominal_conflicts=0)
    used = active.sum(0, keepdims=True) if aggregate else active
    return project(nominal, np.vstack([learning_sum, used]))


def local_project(learning, requests):
    output = []; diagnostics = []
    for l, u in zip(learning, requests):
        if np.linalg.norm(u)>1e-14:
            d, diag = project(l+u, np.stack([l,u]))
        else:
            d, diag = l+u, dict(status='no_request', correction_ratio=0., nominal_conflicts=0)
        output.append(d); diagnostics.append(diag)
    return np.asarray(output), diagnostics


def checks():
    x = np.array([2.,2.]); r = np.array([1.,1.])
    a, _ = global_project(r, np.array([[1.,0.],[0.,1.]]))
    b, info = global_project(r, np.array([[4.,-1.],[-3.,2.]]))
    assert np.array_equal(a,x) and not np.allclose(a,b)
    assert np.allclose(b,np.array([20/13,30/13]),atol=1e-6)
    assert info['minimum_alignment'] >= -2e-6
    z, _ = project(np.array([-1.,2.,3.]), np.array([[1.,0.,0.]]))
    assert np.allclose(z,[0.,2.,3.],atol=1e-6)
    stalled, _ = project(np.array([1.,0.]),np.array([[1.,0.],[-1.,0.]]))
    assert np.linalg.norm(stalled)<1e-6
    again, _ = project(b,np.array([r,[4.,-1.],[-3.,2.]]))
    assert np.allclose(again,b,atol=1e-6)
    # This identity guards against falsely interpreting aggregation as corruption.
    rng=np.random.default_rng(88);l=rng.normal(size=(10,32));u=rng.normal(size=(10,32))
    assert np.allclose((l+u).sum(0),l.sum(0)+u.sum(0))
    return dict(passed=True,same_sum_different_individual_projection=True,
        analytic_halfspace_and_noise_perpendicular=True,opposite_gradients_stall=True,
        idempotent=True,linear_aggregation_identity=True)

if __name__=='__main__':
    import json
    print(json.dumps(checks()))
