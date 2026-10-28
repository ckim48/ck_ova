"""Validate bandwidth physics, zero-change controls, plot data and artifacts."""
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
import downlink_bandwidth_report as report
import bandwidth_tables


def main():
    root, dest, read, sha = report.ROOT, report.DEST, report.read, report.sha
    manifest = read(dest / 'data/manifest.json')
    rows = read(dest / 'data/metrics.json')
    audit = read(dest / 'data/audit.json')
    protocol = read(dest / 'data/protocol.json')
    series = read(dest / 'data/plot_series.json')
    assert manifest['complete'] and len(manifest['figures']) == 8
    assert manifest['bar_bandwidths_mhz'] == [1, 5]
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
    assert len(series) == 56
    for s in series:
        bw = report.study.VALUES if s['kind'] == 'line' else [s['bandwidth_mhz']]
        expected = [indexed[(v, s['dataset'], s['method'])][s['metric']] * s['scale'] for v in bw]
        assert s['y'] == expected
        if s['kind'] == 'line':
            assert s['x'] == [1, 2, 3, 4, 5]
        else:
            assert s['bandwidth_mhz'] in [1, 5] and s['hatch']
            offset = -.185 if s['bandwidth_mhz'] == 1 else .185
            assert np.allclose(s['x'], [s['group_index']+offset])
    for figure in ['04_energy_bar_1x2', '05_delay_bar_1x2']:
        for ds in report.DATASETS:
            for method in report.METHODS:
                group = [s for s in series if (s['figure'], s['dataset'], s['method']) == (figure, ds, method)]
                assert [s['bandwidth_mhz'] for s in group] == [1, 5]
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
                    assert im.width > (1200 if f.get('kind') == 'table' else 1800)
                    assert im.height > (450 if f.get('kind') == 'table' else 800)
            if path.suffix == '.svg' and f.get('kind') not in ['map', 'table']:
                text = path.read_text()
                assert 'Proposed (' not in text
                for title in report.METHODS.values():
                    assert title in text
                if '_bar_' in f['name']:
                    assert panel['y']['limits'][0] == 0
                    assert '1 MHz' in text and '5 MHz' in text
    m = audit['map']
    assert m['passed'] and math.isclose(m['linear_scale'], .72)
    assert m['previous_linear_scale'] == .9 and m['scale_relative_to_previous'] == .8
    assert np.allclose(m['page_size_points'], np.array(m['previous_page_size_points'])*.8)
    assert np.allclose(m['page_size_points'], np.array(m['original_page_size_points'])*.72)
    old_svg = (root / 'results/paper_figures_20261026/mission_map_1x2.svg').read_text()
    new_svg = (dest / '06_mission_map_1x2.svg').read_text()
    pattern = r'(<svg\b[^>]*\b)width="[\d.]+pt" height="[\d.]+pt"'
    assert re.sub(pattern, r'\1SIZE', old_svg) == re.sub(pattern, r'\1SIZE', new_svg)
    assert audit['tables']['passed'] and len(audit['tables']['tables']) == 2
    for name, columns in [('07_performance_table', bandwidth_tables.BASE_COLUMNS),
                          ('08_bandwidth_table', bandwidth_tables.COST_COLUMNS)]:
        table = read(dest / 'data' / (name + '.json'))
        assert table['columns'] == columns and len(table['rows']) == 8
        tex = (dest / (name + '.tex')).read_text()
        assert r'\multirow{4}' in tex and r'\toprule' in tex and r'\bottomrule' in tex
        assert r'\pm' not in tex
        for ds in report.DATASETS:
            block = [r for r in table['rows'] if r['dataset'] == ds]
            for index, col in enumerate(columns):
                displayed = []
                for row in block:
                    actual = indexed[(col['bandwidth'], ds, row['method'])][col['source']]*col['scale']
                    cell = row['cells'][index]
                    assert cell['value'] == actual
                    assert cell['formatted'] == f"{actual:.{col['decimals']}f}"
                    displayed.append(float(cell['formatted']))
                    representation = (r'\textbf{' + cell['formatted'] + '}') if cell['bold'] else cell['formatted']
                    assert representation in tex
                best = (max if col['higher'] else min)(displayed)
                assert [row['cells'][index]['bold'] for row in block] == [v == best for v in displayed]
        item = next(t for t in audit['tables']['tables'] if t['name'] == name)
        assert item['source_metrics_sha256'] == sha(dest / 'data/metrics.json')
    for group in [manifest['source_sha256'], protocol['source_sha256'], m['source_sha256']]:
        for name, digest in group.items():
            assert sha(root / name) == digest, name
    result = dict(passed=True, figures=8, tables=2, arms=40, fresh_arms=24, reused_arms=16,
                  exact_model_controls=8, original_rate_results_unchanged=True,
                  derived_bandwidth_rates_and_costs_verified=True,
                  plot_values_match_metrics=True, regular_endpoint_ticks=True,
                  paired_bars_bandwidths_mhz=[1, 5], tables_match_metrics_and_ties_verified=True,
                  vector_map_uniform_scale=.72, map_scale_relative_to_previous=.8,
                  source_sha256={str(Path(__file__).relative_to(root)): sha(Path(__file__))})
    report.write(dest / 'data/artifact_checks.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
