"""Request-wise minimum-distance control. Directional constraints are proxies."""
import itertools
import numpy as np


def project_rows(center, rows, lower):
    """Solve min .5||d-center||^2 subject to rows @ d >= lower."""
    gram = rows @ rows.T
    gap = lower - rows @ center
    scale = max(np.linalg.norm(center), np.linalg.norm(lower), 1e-5)
    tol = 2e-9 * scale
    best = None
    for mask in itertools.product((False, True), repeat=len(rows)):
        active = np.flatnonzero(mask)
        dual = np.zeros(len(rows))
        if len(active):
            sub = gram[np.ix_(active, active)]
            if np.linalg.matrix_rank(sub, tol=1e-10) < len(active):
                continue
            dual[active] = np.linalg.solve(sub, gap[active])
        slack = gram @ dual - gap
        if np.min(dual, initial=0) < -tol or np.min(slack, initial=0) < -tol:
            continue
        candidate = center + rows.T @ dual
        score = float((candidate-center) @ (candidate-center))
        if best is None or score < best[0]:
            best = score, candidate, np.maximum(dual, 0), active
    if best is None:
        raise ValueError('No feasible projection found; do not silently drop a request.')
    _, candidate, dual, active = best
    assert np.min(rows @ candidate-lower, initial=0) >= -tol*2
    return candidate, dual, active


def problem(learning, anchors, gradient, alpha=1.):
    norms = np.linalg.norm(anchors, axis=1)
    if np.any(norms <= 1e-15) or np.linalg.norm(gradient) <= 1e-15:
        raise ValueError('Zero request anchor or calibration gradient: explicit handling required.')
    rows = np.vstack([anchors/norms[:, None], -gradient/np.linalg.norm(gradient)])
    lower = np.r_[alpha*norms, rows[-1] @ learning]
    center = learning + anchors.sum(axis=0)
    return center, rows, lower, norms


def control(learning, anchors, gradient, alpha=1.):
    center, rows, lower, norms = problem(learning, anchors, gradient, alpha)
    update, dual, active = project_rows(center, rows, lower)
    weights = 1 + dual[:-1]/norms
    correction = -dual[-1]*gradient/np.linalg.norm(gradient)
    assert np.allclose(update, learning + weights @ anchors + correction, atol=1e-11)
    info = dict(alpha=alpha, weights=weights.tolist(), dual=dual.tolist(), active=active.tolist(),
                correction_norm=float(np.linalg.norm(update-center)), update_norm=float(np.linalg.norm(update)),
                raw_update_norm=float(np.linalg.norm(center)),
                request_directional_progress=(anchors @ update/norms**2).tolist(),
                retention_directional_change=float(gradient @ (update-learning)),
                normalized_constraint_slack=(rows @ update-lower).tolist())
    return update, info


def sensitivity_bound(anchors, gradient):
    """Max active-set operator norm for fixed anchors/g and retention RHS Nret@A.

    Valid for full-row-rank rows (hence a continuous piecewise affine map).
    This is a norm sensitivity bound, not a model-quality guarantee.
    """
    norms = np.linalg.norm(anchors, axis=1)
    rows = np.vstack([anchors/norms[:, None], -gradient/np.linalg.norm(gradient)])
    if np.linalg.matrix_rank(rows, tol=1e-10) != len(rows):
        raise ValueError('Dependent constraint rows need a separate sensitivity analysis.')
    basis, _ = np.linalg.qr(rows.T, mode='reduced')
    bound = 1.
    for mask in itertools.product((False, True), repeat=len(rows)):
        active = np.flatnonzero(mask)
        if not len(active):
            continue
        n = rows[active]
        rhs = np.zeros_like(n)
        rhs[active == len(rows)-1] = rows[-1]
        jacobian = np.eye(basis.shape[1]) + (basis.T@n.T) @ np.linalg.solve(n@n.T, (rhs-n)@basis)
        bound = max(bound, float(np.linalg.norm(jacobian, 2)))
    return bound


def test():
    import cvxpy as cp
    rng = np.random.default_rng(841061)
    largest_error = 0.; largest_sensitivity_ratio = 0.
    for trial in range(40):
        learning = rng.normal(size=11)*.01
        anchors = rng.normal(size=(2, 11))*.01
        gradient = rng.normal(size=11)
        center, rows, lower, _ = problem(learning, anchors, gradient)
        actual, _ = control(learning, anchors, gradient)
        value = cp.Variable(11)
        solve = cp.Problem(cp.Minimize(cp.sum_squares(value-center)), [rows@value >= lower])
        solve.solve(solver='CLARABEL', tol_gap_abs=1e-12, tol_feas=1e-12, tol_gap_rel=1e-12)
        assert solve.status == 'optimal'
        error = np.linalg.norm(value.value-actual)
        largest_error = max(largest_error, float(error))
        assert error < 1e-6
        bound = sensitivity_bound(anchors, gradient)
        for _ in range(5):
            perturbation = rng.normal(size=11)*rng.uniform(1e-5, .03)
            changed, _ = control(learning+perturbation, anchors, gradient)
            ratio = np.linalg.norm(changed-actual)/np.linalg.norm(perturbation)
            largest_sensitivity_ratio = max(largest_sensitivity_ratio, float(ratio/bound))
            assert ratio <= bound*(1+1e-7)
    try:
        project_rows(np.zeros(2), np.array([[1., 0.], [-1., 0.]]), np.ones(2))
        raise AssertionError('Infeasible opposite requests were accepted.')
    except ValueError:
        pass
    return dict(status='passed', random_qp_cases=40, sensitivity_pairs=200,
                max_qp_difference=largest_error, max_ratio_to_sensitivity_bound=largest_sensitivity_ratio,
                incompatible_request_case='correctly_reported_infeasible')


if __name__ == '__main__':
    import json
    print(json.dumps(test()))
