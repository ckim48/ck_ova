"""Validate bandwidth physics, zero-change controls, plot data and artifacts."""
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import downlink_bandwidth_report as report
import bandwidth_tables
import aggregation_mse_v1 as mse_study


def main():
    root, dest, read, sha = report.ROOT, report.DEST, report.read, report.sha
    manifest = read(dest / 'data/manifest.json')
    rows = read(dest / 'data/metrics.json')
    audit = read(dest / 'data/audit.json')
    protocol = read(dest / 'data/protocol.json')
    series = read(dest / 'data/plot_series.json')
    assert manifest['complete'] and len(manifest['figures']) == 9
    assert manifest['total_unique_arms'] == 72 and manifest['mse_new_arms'] == 32
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
    mse_rows = read(dest / 'data/mse_metrics.json')
    mse_audit = read(dest / 'data/mse_audit.json')
    mse_protocol = read(dest / 'data/mse_protocol.json')
    assert len(mse_rows) == 40 and mse_audit['passed']
    assert sum(r['origin'] == 'fresh_mse_execution' for r in mse_rows) == 32
    mse_index = {(r['mse_threshold'], r['dataset'], r['method']): r for r in mse_rows}
    for r in mse_rows:
        assert r['relative_rmse_coefficient'] == math.sqrt(r['mse_threshold'])
        assert r['FL_rounds'] == 200 and r['FU_cycles'] == 225
        assert r['max_mse_budget_utilization'] <= 1+6e-5
        assert math.isclose(r['total_energy_j'], r['communication_j']+r['compute_j']+r['hover_j'], rel_tol=1e-12)
        if r['mse_threshold'] == .04:
            old = indexed[(1, r['dataset'], r['method'])]
            for key in ['total_energy_j', 'delay_s', 'retained_mR_percent', 'retained_loss', 'forget_KL']:
                assert math.isclose(r[key], old[key], rel_tol=1e-12, abs_tol=1e-8)
    assert len(series) == 64
    for s in series:
        if s.get('axis') == 'aggregation_mse':
            assert s['x'] == mse_study.VALUES
            assert s['y'] == [mse_index[(v, s['dataset'], s['method'])][s['metric']]*s['scale'] for v in s['x']]
            continue
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
                visible_text = ''.join(''.join(ET.parse(path).getroot().itertext()).split())
                for title in report.METHODS.values():
                    assert title.replace(' ', '') in visible_text
                if '_bar_' in f['name']:
                    assert panel['y']['limits'][0] == 0
                    assert '1 MHz' in text and '5 MHz' in text
    m = audit['map']
    assert m['passed'] and math.isclose(m['linear_scale'], .576)
    assert m['previous_linear_scale'] == .72 and m['scale_relative_to_previous'] == .8
    assert np.allclose(m['map_panel_size_inches'], np.array(m['previous_map_panel_size_inches'])*.8)
    assert np.allclose(m['map_panel_size_inches'], np.array(m['original_map_panel_size_inches'])*.576)
    assert m['tick_fontsize_pt'] == m['visit_fontsize_pt'] == 12 and m['labels_inside_canvas']
    if m.get('location_revision') == 'yeouido_v1':
        assert not m['geometry_and_snapshot_heads_unchanged'] and m['new_geometry_verified']
        geometry = read(root/m['geometry_checks'])
        assert geometry['passed'] and geometry['satellite_and_overlay_registered']
        assert geometry['iot_not_on_water'] and geometry['ten_uav_identities_preserved']
        assert geometry['no_FL_FU_retraining']
    else:
        assert m['geometry_and_snapshot_heads_unchanged']
    ns = {'s': 'http://www.w3.org/2000/svg'}
    new_svg = ET.parse(dest / '06_mission_map_1x2.svg').getroot()
    reference_svg = ET.parse(dest / '01_downlink_bandwidth_energy_1x2.svg').getroot()
    def font_css(text):
        properties = dict(part.strip().split(':', 1) for part in text.get('style').split(';') if ':' in part)
        return properties['font'].strip()
    reference_font = font_css(reference_svg.find('.//s:g[@id="xtick_1"]//s:text', ns))
    labels = new_svg.findall('.//s:text', ns)
    assert len(labels) == 25
    assert all(font_css(t) == reference_font for t in labels)
    rectangles = new_svg.findall('.//s:clipPath/s:rect', ns)
    assert len(rectangles) == 2
    assert np.allclose([[float(r.get('width')), float(r.get('height'))] for r in rectangles], np.array(m['map_panel_size_inches'])*72)
    assert audit['tables']['passed'] and len(audit['tables']['tables']) == 2
    for name, columns in [('07_performance_table', bandwidth_tables.BASE_COLUMNS),
                          ('08_bandwidth_table', bandwidth_tables.COST_COLUMNS)]:
        table = read(dest / 'data' / (name + '.json'))
        assert table['columns'] == columns and len(table['rows']) == 8
        tex = (dest / (name + '.tex')).read_text()
        assert r'\multirow{4}' in tex and r'\toprule' in tex and r'\bottomrule' in tex
        assert r'\pm' not in tex
        if name == '07_performance_table':
            assert 'Forget KL' not in tex and all(c['key'] != 'forget_kl' for c in columns)
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
    for group in [manifest['source_sha256'], protocol['source_sha256'], m['source_sha256'], mse_protocol['source_sha256']]:
        for name, digest in group.items():
            assert sha(root / name) == digest, name
    result = dict(passed=True, figures=9, tables=2, arms=40, fresh_arms=24, reused_arms=16,
                  mse_arms=40, fresh_mse_arms=32, total_unique_arms=72,
                  exact_model_controls=8, original_rate_results_unchanged=True,
                  derived_bandwidth_rates_and_costs_verified=True,
                  plot_values_match_metrics=True, regular_endpoint_ticks=True,
                  paired_bars_bandwidths_mhz=[1, 5], tables_match_metrics_and_ties_verified=True,
                  map_panel_scale=.576, map_panel_scale_relative_to_previous=.8,
                  map_location_revision=m.get('location_revision','daejeon'),
                  map_text_matches_chart_ticks_pt=12, table07_forget_KL_removed=True,
                  source_sha256={str(Path(__file__).relative_to(root)): sha(Path(__file__))})
    report.write(dest / 'data/artifact_checks.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
