"""Check optimizer-state handover, actual waveform costs and transport limits."""
import numpy as np
import adaptive_iq_physical_v1 as s
import paper_uav_charts as ledger


def main():
    s.parent.learning.quality.infra.PHYSICAL = ledger.physical
    settings = s.read(s.ROOT/'results/combined_iq_experiment_a_20261030/iq/cases/baseline/settings.json')
    ledger.physical.PARAMETERS = settings['energy_model']
    path = s.ROOT/'results/combined_iq_diagnostic_v1_20261030/snapshots/capera/joint18.npz'
    z = np.load(path); meta = s.read(path.with_suffix('.json'))
    l, u, channels, ell = [z[k] for k in ['l', 'u', 'channels', 'ell']]
    c = meta['config']; rows = []
    for setting in ['block_slow', 'diagonal_slow']:
        s.SETTING = setting
        for mode in s.METHODS:
            s.STATE.clear()
            optimizer = s.controller.RMS(s.controller.SETTINGS[setting], mode if mode != 'iq_shared' else 'combined')
            optimizer.prepare([1])
            optimizer.apply(l.sum(0), u.sum(0), np.zeros((2, l.shape[1])), .0035)
            optimizer.head = 0
            s.STATE.update(optimizer=optimizer, ids=[1], lr=.0035)
            extra = s.handover(1, channels, c)
            assert extra['bytes'] == optimizer.state_bytes()+128
            assert np.isclose(extra['duration_s'], 8*extra['bytes']/extra['rate_bps'])
            assert np.isclose(extra['RF_j'], c['power_limit_w']*extra['duration_s'])
            module = s.radio_module(mode)
            received, r = module.analog(l, u, channels, c, meta['seed'], ell)
            assert received is not None
            assert np.isclose(np.linalg.norm(received), r['optimizer']['raw_norm'])
            assert r['energy_bound_j']+r['state_handover']['RF_j'] <= .1*(1+2e-6)
            assert r['uplink_s']+r['state_handover']['duration_s'] <= c['round_uplink_deadline_s']+1e-12
            rho, _, _, _ = module.inputs(l, u, c, ell)
            x = np.array(r['transceiver_x']); head = r['head']
            h = np.abs(channels[:, head]).copy(); h[head] = 1.
            costs = ((rho**2/h[None,:,None]**2)*x[:,None,:]).sum(0)
            costs[head] = 0.
            assert np.allclose(costs.sum(1), r['user_RF_energy_j'], rtol=1e-10, atol=1e-18)
            assert all(b <= a*(1+1e-8) for a,b in zip(r['BCD_objectives'],r['BCD_objectives'][1:]))
            rows.append(dict(setting=setting, mode=mode, head=head, state_bytes=optimizer.state_bytes(),
                conditional_handover_bytes=extra['bytes'], conditional_handover_s=extra['duration_s'],
                actual_handover=r['state_handover'], RF_j=r['energy_bound_j']))
    s.write(s.OUT/'physical_checks.json',dict(passed=True, math=s.controller.checks(), cases=rows))
    print('Passed',len(rows),'physical state/waveform cases',flush=True)


if __name__=='__main__': main()
