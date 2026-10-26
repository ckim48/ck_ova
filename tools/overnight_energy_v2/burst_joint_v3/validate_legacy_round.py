"""Verify the training adapter against the independent full-size radio audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args(); root = args.project.resolve()
    sys.path[:0] = [str(root), str(root/'tools/overnight_energy_v2'), str(root/'tools/overnight_energy_v2/joint_power_v3')]
    import burst_transport as adapter
    from run_variant import source_hashes
    from adaptive_precision import exact_statistics
    from transport import energy_accounting, recover
    source = root/'results/failed_round_diagnostic'
    config = dict(json.loads((source/'metadata.json').read_text())['config'], deletion_relative_tolerance=.6)
    input_sha = hashlib.sha256((source/'updates.npz').read_bytes()).hexdigest()
    with np.load(source/'updates.npz') as data:
        learning, unlearning, channels = [data[k].copy() for k in ('learning','unlearning','channels')]
    audit = root/'results/overnight_energy_v2/joint_power_v3'
    manifest = json.loads((audit/'manifest.json').read_text())
    assert manifest['updates_sha256'] == input_sha
    assert manifest['variant_sha256'] == hashlib.sha256((root/'tools/overnight_energy_v2/joint_power_v3/wireless_joint_v3.py').read_bytes()).hexdigest()
    old = json.loads((audit/'results.json').read_text()); rows = []
    for mode in adapter.MODES:
        started = time.monotonic(); design, inputs, meta = adapter.design(learning, unlearning, channels, config, mode)
        assert design is not None, mode
        expected_mode = 'orthogonal' if mode == 'phase_grid' else mode
        previous = [r for r in old if r['bits']==8 and r['relative_tolerance']==.6
                    and r['mode']==expected_mode and r['status']=='feasible']
        expected = min(r['energy_bound_j'] for r in previous)
        stats = energy_accounting(design, config)
        assert np.isclose(stats['energy_bound_j'], expected, rtol=1e-5, atol=1e-10), (mode, stats, expected)
        stats.update(exact_statistics(design, *inputs[:3], learning.shape[1]))
        if mode == 'mixed_aircomp':
            stats['unlearning_component_error'] = None
        assert stats['analytic_total_rmse'] <= stats['error_bound']*(1+1e-4)
        assert stats['error_bound'] <= meta['epsilon']*(1+1e-5)
        assert stats['expected_energy_random_codewords_j'] <= stats['energy_bound_j']*(1+1e-6)
        assert np.isclose(stats['uplink_s'], inputs[-1].sum()/config['symbol_rate_hz'])
        ideal = (learning+unlearning).sum(axis=0); generator = np.random.default_rng(20261021)
        mse = np.array([np.sum((recover(design, *inputs[:3], generator)[:len(ideal)]-ideal)**2) for _ in range(64)])
        expected_mse = stats['analytic_total_rmse']**2
        assert abs(mse.mean()-expected_mse) <= 6*mse.std(ddof=1)/np.sqrt(len(mse))+expected_mse*1e-4
        rows.append(dict(mode=mode, expected_standalone_energy_j=expected,
            empirical_mse=float(mse.mean()), analytic_mse=expected_mse, **stats, **meta,
            seconds=time.monotonic()-started))
        print(json.dumps(rows[-1]), flush=True)
    output = root/'results/overnight_energy_v2/burst_joint_v3'
    output.mkdir(parents=True, exist_ok=True)
    report = {'status':'passed', 'source_sha256':source_hashes(root), 'updates_sha256':input_sha,
              'validation_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'rows':rows, 'limits':'Radio adapter equivalence and conditional Gaussian error check; no new VLM or forgetting guarantee.'}
    (output/'legacy_adapter_validation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
