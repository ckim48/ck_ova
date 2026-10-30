"""Independent physical checks for both representations in experiment A."""
from pathlib import Path
import numpy as np
import combined_iq_experiment_a as study


def main():
    c = dict(block_dimension=128, relative_tolerance=.2, absolute_tolerance_floor=1e-8,
        symbol_rate_hz=1e6, power_limit_w=.32, available_energy_j=.5,
        round_uplink_deadline_s=1.31072, coherence_time_s=.01, max_ao_iterations=8)
    rng = np.random.default_rng(718)
    l = rng.normal(size=(10, 512))*.001
    u = rng.normal(size=l.shape)*.0003
    h = np.full((10, 10), 3.)
    np.fill_diagonal(h, 0)
    channels = h*np.exp(1j*rng.uniform(-np.pi, np.pi, h.shape))
    ell = np.array([[1., 2., 3., 4.], [4., 1., 2., 3.]])
    results = []
    budgets = []
    for mode in study.METHODS:
        radio = study.radio_module(mode)
        best, audit = radio.design(l, u, channels, c, ell)
        assert best is not None
        budgets.append((np.sum(np.square(audit['component_epsilon'])), audit['normalized_distortion_limits']))
        head, dm, n, W = best['head'], 128, best['n'], c['symbol_rate_hz']
        remote = np.arange(10) != head
        for block in range(4):
            sl = slice(block*dm, (block+1)*dm)
            x = best['x'][:, block]
            cr = 1/np.sqrt(W*x)
            energy = np.zeros(10)
            signal = np.zeros(dm, complex)
            for user in np.flatnonzero(remote):
                b = np.sqrt(W*x)/channels[user, head]
                wave = (b[0]*(l[user, sl]+u[user, sl]) if mode == 'combined'
                        else b[0]*l[user, sl] + 1j*b[1]*u[user, sl])
                expanded = np.tile(wave, n)/np.sqrt(n)
                energy[user] = np.vdot(expanded, expanded).real/W
                signal += channels[user, head]*wave
            assert np.allclose(energy, best['energy'][:, block], rtol=1e-10, atol=1e-14)
            recovered = cr[0]*signal.real
            if mode == 'iq':
                assert np.allclose(recovered, l[remote, sl].sum(0), atol=1e-12)
                assert np.allclose(cr[1]*signal.imag, u[remote, sl].sum(0), atol=1e-12)
                recovered += cr[1]*signal.imag
            assert np.allclose(recovered, (l+u)[remote, sl].sum(0), atol=1e-12)
        err = rng.normal(size=(4000, 2, 4, dm))*np.sqrt(best['variance'][None, :, :, None]/dm)
        observed = np.mean(np.sum(err**2, axis=(2, 3)), axis=0)
        assert np.allclose(observed, best['mse'], rtol=.012, atol=1e-16)
        results.append(dict(method=mode, passed=True, exact_complex_waveform_recovery=True,
            joint_power_verified=True, analytic_MSE=best['mse'].tolist(),
            observed_MSE=observed.tolist(), trials=4000))
    assert np.isclose(budgets[0][0], budgets[1][0])
    assert np.allclose(budgets[0][1], budgets[1][1])
    # Both representations must be identical when only one task is active.
    z = np.zeros_like(l)
    r0, m0 = study.radio_module('combined').analog(l, z, h, c, 11, ell)
    r1, m1 = study.radio_module('iq').analog(l, z, h, c, 11, ell)
    assert np.allclose(r0, r1, atol=1e-14)
    assert np.isclose(m0['energy_bound_j'], m1['energy_bound_j'])
    # The combined representation must preserve true cancellation; no norm-bound inflation.
    rc, mc = study.radio_module('combined').analog(l, -l, h, c, 11, ell)
    assert np.all(rc == 0) and mc['energy_bound_j'] == mc['uplink_s'] == 0
    result = dict(passed=True, shared_joint_MSE_budget=True, shared_task_distortion_budget=True,
        single_task_equivalence=True, true_combined_cross_term=True, methods=results,
        source_sha256=study.sha(Path(__file__)))
    study.write(study.OUT/'physical_checks.json', result)
    print(result)


if __name__ == '__main__':
    main()
