"""Independent waveform/cost check on two pre-existing frozen real updates."""
import numpy as np
import head_fusion_study_v1 as s


def main():
    s.register()
    rows = []
    for ds in ['capera', 'airscape']:
        path = s.ROOT/'results/combined_iq_diagnostic_v1_20261030/snapshots'/ds/'joint36.npz'
        z = np.load(path)
        meta = s.read(path.with_suffix('.json'))
        raw_l, raw_u, channels, ell = [z[k] for k in ['l', 'u', 'channels', 'ell']]
        c, seed = meta['config'], meta['seed']
        expected_eps = s.parent.iq.inputs(raw_l, raw_u, c, ell)[1]
        for mode in s.METHODS:
            s.FAMILY, s.MODE = 'current', mode
            s.RAW.clear()
            s.RAW.update(l=raw_l, u=raw_u, local=[])
            l, u = raw_l, raw_u
            if mode == 'combined_local_pc':
                l, u, s.RAW['local'] = s.fusion.local_fuse(l, u)
            module = s.radio_module(mode)
            received, r = module.analog(l, u, channels, c, seed, ell)
            assert received is not None
            var = np.array(r['block_variance'])
            m, dm = var.shape[1], c['block_dimension']
            noise = np.random.default_rng(seed).normal(size=(2, m, dm))*np.sqrt(var[:, :, None]/dm)
            expected = (l+u).sum(0)+noise.sum(0).ravel()
            if mode in ['iq_head_pc', 'iq_head_nr']:
                lh, uh = l.sum(0)+noise[0].ravel(), u.sum(0)+noise[1].ravel()
                expected = s.fusion.fuse(lh, uh, var, mode == 'iq_head_nr')[0]
            assert np.allclose(received, expected, rtol=1e-10, atol=1e-14)
            rho, eps, _, _ = module.inputs(l, u, c, ell)
            if mode.startswith('combined'):
                assert np.isclose(eps[0]**2, (expected_eps**2).sum(), rtol=1e-12)
            else:
                assert np.array_equal(eps, expected_eps)
            x = np.array(r['transceiver_x'])
            head = r['head']
            # Independently rebuild actual RF costs, including the combined cross term.
            h = np.abs(channels[:, head]).copy()
            h[head] = 1.
            energy = ((rho**2/h[None, :, None]**2)*x[:, None, :]).sum(0)
            energy[head] = 0.
            assert np.allclose(energy.sum(1), r['user_RF_energy_j'], rtol=1e-10, atol=1e-18)
            assert np.allclose(energy*c['symbol_rate_hz']/(r['frame_count']*dm),
                r['user_block_power_w'], rtol=1e-10, atol=1e-18)
            rows.append(dict(dataset=ds, method=mode, input_sha256=s.sha(path),
                received_max_abs_error=float(np.max(np.abs(received-expected))),
                RF_energy_j=float(energy.sum()), raw_budget_preserved=True,
                head_corrected=r.get('head_fusion', {}).get('corrected', False)))
    # Same combined sum does not identify the non-linear head-fusion result.
    a = s.fusion.fuse(np.array([.5, 0.]), np.array([.5, 0.]))[0]
    b = s.fusion.fuse(np.array([0., 1.]), np.array([1., -1.]))[0]
    assert not np.allclose(a, b)
    s.write(s.OUT/'physical_checks.json', dict(passed=True, cases=rows,
        identical_sum_counterexample=dict(combined_sum=[1., 0.], first_fused=a.tolist(), second_fused=b.tolist()),
        scope='Frozen-update integration check, not additional independent training trials.'))
    print('Passed', len(rows), 'frozen-update waveform/cost checks', flush=True)


if __name__ == '__main__':
    main()
