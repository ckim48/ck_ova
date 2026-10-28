"""Validate bandwidth physics, zero-change controls, plot data and artifacts."""
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
import downlink_bandwidth_report as report


def main():
    root, dest, read, sha = report.ROOT, report.DEST, report.read, report.sha
    manifest = read(dest / 'data/manifest.json')
    rows = read(dest / 'data/metrics.json')
    audit = read(dest / 'data/audit.json')
    protocol = read(dest / 'data/protocol.json')
    series = read(dest / 'data/plot_series.json')
    assert manifest['complete'] and len(manifest['figures']) == 6
    assert manifest['methods'] == report.METHODS
    assert len(rows) == 40 and audit['passed']
    assert len(audit['baseline_controls']) == 8
    assert all(all(c[k] for k in ['identical_model_and_quality', 'identical_transaction_order_and_radio', 'clock_reproduced']) for c in audit['baseline_controls'])
    assert sum(r['origin'] == 'fresh_bandwidth_execution' for r in rows) == 24
    assert sum(r['origin'] == 'equivalent_rate_reuse' for r in rows) == 16
    indexed = {(r['bandwidth_mhz'], r['dataset'], r['method']): r for r in rows}
    old_rows = read(root / 'final_final_charts/proposed_pks/data/metrics.json')
    for r in rows:
        assert r['FL_rounds'] == 200 and r['FU_cycles'] == 225
        assert r['downlink_rate_bps'] == r['bandwidth_mhz'] * 1e6 * r['spectral_efficiency_bps_hz']
        assert math.isclose(r['total_energy_j'], sum(r[k] for k in ['communication_j', 'compute_j', 'hover_j']), rel_tol=1e-12)
        if r['bandwidth_mhz'] in [1, 2, 4]:
            case = 'baseline' if r['bandwidth_mhz'] == 1 else f"downlink_{r['bandwidth_mhz']}"
            old = next(o for o in old_rows if (o['case'], o['dataset'], o['method']) == (case, r['dataset'], r['method']))
            for key in ['total_energy_j', 'delay_s', 'mission_delay_s', 'retained_mR_percent', 'retained_loss', 'forget_KL']:
                assert math.isclose(old[key], r[key], rel_tol=1e-12, abs_tol=1e-8), (r['case'], r['method'], key)
    assert len(series) == 40
    for s in series:
        bw = report.study.VALUES if s['kind'] == 'line' else [s['bandwidth_mhz']]
        expected = [indexed[(v, s['dataset'], s['method'])][s['metric']] * s['scale'] for v in bw]
        assert s['y'] == expected
        if s['kind'] == 'line':
            assert s['x'] == [1, 2, 3, 4, 5]
        else:
            assert s['bandwidth_mhz'] == 1 and s['hatch']
    for f in manifest['figures']:
        precisions = []
        for panel in f['axes']:
            digits = {}
            for dim in ['x', 'y']:
                axis = panel[dim]
                if axis.get('kind') == 'categorical':
                    continue
                ticks = axis['ticks']
                assert axis['limits'] == [ticks[0], ticks[-1]]
                assert np.allclose(np.diff(ticks), ticks[1]-ticks[0])
                precision = [len(s.split('.')[-1]) if '.' in s else 0 for s in axis['labels']]
                assert len(set(precision)) == 1
                digits[dim] = precision[0]
            precisions.append(digits)
        if precisions:
            assert precisions[0] == precisions[1]
        for item in f['files']:
            path = dest / item['file']
            assert sha(path) == item['sha256']
            if path.suffix == '.png':
                with Image.open(path) as im:
                    assert im.width > 1800 and im.height > 800
            if path.suffix == '.svg' and f.get('kind') != 'map':
                text = path.read_text()
                assert 'Proposed (' not in text
                for title in report.METHODS.values():
                    assert title in text
                if '_bar_' in f['name']:
                    assert panel['y']['limits'][0] == 0
    m = audit['map']
    assert m['passed'] and m['linear_scale'] == .9
    assert np.allclose(m['page_size_points'], np.array(m['original_page_size_points'])*.9)
    old_svg = (root / 'results/paper_figures_20261026/mission_map_1x2.svg').read_text()
    new_svg = (dest / '06_mission_map_1x2.svg').read_text()
    pattern = r'(<svg\b[^>]*\b)width="[\d.]+pt" height="[\d.]+pt"'
    assert re.sub(pattern, r'\1SIZE', old_svg) == re.sub(pattern, r'\1SIZE', new_svg)
    for group in [manifest['source_sha256'], protocol['source_sha256'], m['source_sha256']]:
        for name, digest in group.items():
            assert sha(root / name) == digest, name
    result = dict(passed=True, figures=6, arms=40, fresh_arms=24, reused_arms=16,
                  exact_model_controls=8, original_rate_results_unchanged=True,
                  derived_bandwidth_rates_and_costs_verified=True,
                  plot_values_match_metrics=True, regular_endpoint_ticks=True,
                  vector_map_uniform_scale=.9,
                  source_sha256={str(Path(__file__).relative_to(root)): sha(Path(__file__))})
    report.write(dest / 'data/artifact_checks.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
