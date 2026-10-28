"""Check identities, original measurements, cost components and plot arrays."""
from pathlib import Path
import math

import numpy as np
from PIL import Image
import proposed_pks_report as report


def main():
    root, dest = report.ROOT, report.DEST
    read, sha = report.read, report.sha
    manifest = read(dest / 'data/manifest.json')
    identity = read(dest / 'data/method_identity.json')
    rows = read(dest / 'data/metrics.json')
    series = read(dest / 'data/plot_series.json')
    budget = read(dest / 'data/budget400.json')
    audit = read(dest / 'data/audit.json')
    assert manifest['complete'] and len(manifest['figures']) == 14
    assert manifest['methods'] == report.METHODS
    assert identity['proposed_source_variant'] == 'gate_select16'
    assert not identity['original_adaptive_proposed_included'] and not identity['PKB25_substituted']
    assert len(rows) == 104 and len(budget) == 8
    assert set(r['method'] for r in rows) == set(report.METHODS)
    indexed = {(r['case'], r['dataset'], r['method']): r for r in rows}
    assert len(indexed) == len(rows)
    previous = read(root / 'final_final_charts/four_methods/data/metrics.json')
    old_index = {(r['case'], r['dataset'], r['method']): r for r in previous}
    diagnosis = read(root / 'final_final_charts/pks_kl_repair/data/metrics.json')
    for row in rows:
        key = row['case'], row['dataset'], row['method']
        assert row['label'] == report.METHODS[row['method']]
        assert row['FU_cycles'] == 225
        assert math.isclose(row['total_energy_j'], sum(row[k] for k in ['communication_j', 'compute_j', 'hover_j']), rel_tol=1e-12)
        if row['method'] != 'gate_select16':
            old = old_index[(key[0], key[1], report.TRANSFER_IDS[key[2]])]
            for metric in ['total_energy_j', 'delay_s', 'mission_delay_s', 'retained_mR_percent',
                           'retained_loss', 'forget_KL', 'worst_request_KL']:
                assert row[metric] == old[metric], (key, metric)
        elif row['case'] == 'baseline':
            old = next(r for r in diagnosis if r['dataset'] == row['dataset'] and r['method'] == 'pks' and r['stage'] == 'round200')
            assert row['retained_mR_percent'] == old['retained_mR_percent']
            assert row['retained_loss'] == old['retained_loss']
            assert row['forget_KL'] == old['common_KL']
    for s in series:
        assert s['method'] in report.METHODS and len(s['x']) == len(s['y'])
        assert all(a <= b for a, b in zip(s['x'], s['x'][1:]))
        if s.get('domain') == 'round':
            assert s['x'][-1] == 200
            metric = {'loss': 'retained_loss', 'mR': 'retained_mR_percent', 'KL': 'forget_KL'}[s['metric']]
            assert s['y'][-1] == indexed[('baseline', s['dataset'], s['method'])][metric]
        elif s.get('domain') == 'time':
            assert s['drawstyle'] == 'steps-post' and s['x'][-1] <= 400
            assert s['display_x'][-1] == 400 and s['display_y'][-1] == s['y'][-1]
            b = next(b for b in budget if b['dataset'] == s['dataset'] and b['method'] == s['method'])
            assert s['y'][-1] == b['mR_percent' if s['metric'] == 'mR' else 'loss']
        else:
            assert s['x'] == report.AXES[s['axis']].get('display_values', report.AXES[s['axis']]['values'])
            assert s['y'] == [indexed[(c, s['dataset'], s['method'])][s['metric']] * s['scale'] for c in s['cases']]
    old_time = read(root / 'final_final_charts/same_time/data/metrics.json')
    for b in budget:
        old = next(r for r in old_time if r['dataset'] == b['dataset'] and r['method'] == b['method'] and r['budget_s'] == 400)
        assert b['mR_percent'] == old['retained_mR_percent'] and b['loss'] == old['retained_loss']
        assert b['FL_round'] == old['evaluated_FL_round'] and b['checkpoint_s'] == old['quality_checkpoint_s']
    for figure in manifest['figures']:
        precisions = []
        for panel in figure['axes']:
            precision = {}
            for dimension in ['x', 'y']:
                a = panel[dimension]
                ticks = a['ticks']
                assert a['limits'] == [ticks[0], ticks[-1]]
                assert np.allclose(np.diff(ticks), ticks[1] - ticks[0])
                digits = [len(v.split('.')[-1]) if '.' in v else 0 for v in a['labels']]
                assert len(set(digits)) == 1
                precision[dimension] = digits[0]
            precisions.append(precision)
        assert precisions[0] == precisions[1]
        for item in figure['files']:
            path = dest / item['file']
            assert sha(path) == item['sha256']
            if path.suffix == '.png':
                with Image.open(path) as im:
                    assert im.width > 1800 and im.height > 800
    assert audit['passed'] and len(audit['shared_condition_checks']) == 104
    assert len(audit['continued_time_checks']) == 8
    for group in [manifest['source_sha256'], identity['source_sha256']]:
        for name, digest in group.items():
            assert sha(root / name) == digest, name
    result = dict(passed=True, figures=14, arms=104, fresh_PKS_arms=24,
                  original_PKS_identity_verified=True, original_comparators_unchanged=True,
                  original_same_time_quality_exact=True, energy_components_recomputed=True,
                  raw_series_and_regular_endpoint_axes_verified=True,
                  source_sha256={str(Path(__file__).relative_to(root)): sha(Path(__file__))})
    report.write(dest / 'data/artifact_checks.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
