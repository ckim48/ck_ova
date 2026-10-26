"""Bundle compact summary bars, observed convergence curves, and the larger map.

Reads completed v4 results only. No training, smoothing, or synthetic points.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

import plot_event_mission_reference_style as bars

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/paper_figures_20261026'
MARKERS = ['o', 's', '^', 'D']
LINESTYLES = ['-', '--', '-.', ':']


def curve_data(campaign, protocol):
    samples, inputs, counts = [], {}, []
    summary = bars.read(campaign / 'figures/metrics.json')['rows']
    for dataset in protocol['datasets']:
        folder = campaign / (dataset.lower() + '_seed0')
        initial_file = folder / 'initial20/rounds.json'
        initial = [r for r in bars.read(initial_file) if r.get('evaluation')]
        inputs[str(initial_file.relative_to(ROOT))] = bars.sha(initial_file)
        for method, label in protocol['plotting']['labels'].items():
            series = []
            for row in initial:
                ev = row['evaluation']['retained']
                series.append(dict(dataset=dataset, method=method, label=label,
                    stage='common_initialization', sequence=row['round'],
                    completed_FL_rounds=row['round'], kind='fl',
                    loss=ev['loss'], mR_percent=100 * ev['mR'],
                    source=str(initial_file.relative_to(ROOT))))
            for path in sorted((folder / method / 'transactions').glob('*.json')):
                row = bars.read(path)
                if not row.get('evaluation'):
                    continue
                inputs[str(path.relative_to(ROOT))] = bars.sha(path)
                ev = row['evaluation']['retained']
                series.append(dict(dataset=dataset, method=method, label=label,
                    stage='adaptation', sequence=row['sequence'],
                    completed_FL_rounds=row['completed_FL_rounds'], kind=row['kind'],
                    loss=ev['loss'], mR_percent=100 * ev['mR'],
                    source=str(path.relative_to(ROOT))))
            x = [r['completed_FL_rounds'] for r in series]
            assert len(x) > 2 and x == sorted(x) and x[0] == 5 and x[-1] == 200
            assert all(math.isfinite(r['loss']) and 0 <= r['mR_percent'] <= 100 for r in series)
            final = next(r for r in summary if r['dataset'] == dataset and r['method'] == method)
            assert math.isclose(series[-1]['loss'], final['loss'], abs_tol=1e-12)
            assert math.isclose(series[-1]['mR_percent'], 100 * final['mR'], abs_tol=1e-12)
            counts.append(dict(dataset=dataset, method=method, evaluations=len(series),
                               first_FL_round=x[0], last_FL_round=x[-1],
                               repeated_FL_round_evaluations=len(x) - len(set(x))))
            samples.extend(series)
    return samples, inputs, counts


def plot_curves(output, protocol, samples):
    bars.configure()
    labels = protocol['plotting']['labels']
    results = []
    for key, ylabel, name in [('loss', 'Retained loss', 'loss_convergence_1x2'),
                              ('mR_percent', 'Retained mR (%)', 'mr_convergence_1x2')]:
        fig, axes = plt.subplots(1, 2, figsize=bars.FIGSIZE)
        fig.subplots_adjust(left=.12, right=.985, bottom=.22, top=.87, wspace=.38)
        handles = [Line2D([], [], color=color, linestyle=style, marker=marker,
                          ms=5, mew=1.1, mfc='white', mec=color, lw=1.5, label=label)
                   for color, style, marker, label in
                   zip(bars.COLORS, LINESTYLES, MARKERS, labels.values())]
        legend = fig.legend(handles=handles, ncol=4, loc='upper center',
                            bbox_to_anchor=(.5, 1.0))
        measured = []
        for ax, dataset in zip(axes, protocol['datasets']):
            for index, (method, label) in enumerate(labels.items()):
                series = [r for r in samples if r['dataset'] == dataset and r['method'] == method]
                x = np.array([r['completed_FL_rounds'] for r in series])
                y = np.array([r[key] for r in series])
                line, = ax.plot(x, y, color=bars.COLORS[index], lw=1.5,
                    linestyle=LINESTYLES[index], marker=MARKERS[index], ms=5,
                    mfc='white', mec=bars.COLORS[index], mew=1.1,
                    markevery=(index, 7), label=label)
                assert np.array_equal(line.get_xdata(), x) and np.array_equal(line.get_ydata(), y)
            ax.set_xlim(0, 200)
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_xlabel('FL rounds', labelpad=3)
            ax.set_ylabel(ylabel, labelpad=5)
            ax.set_box_aspect(bars.PANEL_ASPECT)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            ax.margins(y=.09)
            ax.grid(True, linestyle=':')
            ax.set_axisbelow(True)
            ax.tick_params(direction='out', pad=3)
            assert not ax.texts
        fig.canvas.draw()
        for ax, letter, dataset in zip(axes, ['a', 'b'], protocol['datasets']):
            box = ax.get_position()
            fig.text((box.x0 + box.x1) / 2, .02, f'({letter}) {dataset}',
                     ha='center', va='bottom', fontsize=14)
            px = ax.get_window_extent()
            assert math.isclose(px.height / px.width, bars.PANEL_ASPECT, abs_tol=1e-6)
            measured.append(dict(dataset=dataset, ylim=list(ax.get_ylim()),
                                 panel_size_inches=[px.width / fig.dpi, px.height / fig.dpi]))
        assert legend.get_window_extent().x0 >= 0
        assert legend.get_window_extent().x1 <= fig.bbox.width
        files = []
        for suffix in ['png', 'pdf', 'svg']:
            path = output / f'{name}.{suffix}'
            fig.savefig(path, dpi=400, bbox_inches='tight')
            if suffix == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
            files.append(dict(file=path.name, sha256=bars.sha(path)))
        plt.close(fig)
        results.append(dict(metric=key, panels=measured, files=files))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, default=bars.DEFAULT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    campaign, output = args.campaign.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = campaign / 'protocol.json'
    protocol = bars.read(protocol_path)
    registered = {}
    for group in ['source_sha256', 'frozen_source_sha256']:
        for name, expected in protocol[group].items():
            assert bars.sha(ROOT / name) == expected, name
            registered[name] = expected
    env = dict(os.environ, MPLCONFIGDIR='/tmp/ova_mpl_style')
    subprocess.run([sys.executable, str(ROOT / 'tools/plot_event_mission_reference_style.py'),
                    '--campaign', str(campaign), '--output', str(output)],
                   cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    for name in ['metrics.json', 'metrics.csv']:
        source = campaign / 'figures' / name
        if name.endswith('.csv'):
            # Stable LF serialization so published hashes survive Git normalization.
            (output / name).write_text(source.read_text())
        else:
            shutil.copy2(source, output / name)
    samples, inputs, counts = curve_data(campaign, protocol)
    with (output / 'convergence_samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(samples)
    curves = plot_curves(output, protocol, samples)
    bars.save_json(output / 'convergence_manifest.json', dict(
        source_campaign=str(campaign.relative_to(ROOT)),
        x_axis='Completed FL rounds, including the common first 20 rounds',
        sample_policy='All logged evaluations in chronological order, starting at round 5. FU-only evaluations at the same completed-FL count are retained. No fabricated round-0 point, smoothing, resampling or confidence interval. Lines connect measured checkpoints.',
        mR_units='Percent; 100 times the logged fraction',
        final_points_match_summary=True, series=counts, input_sha256=inputs, plots=curves))
    transactions = campaign / 'capera_seed0/proposed_fu_guard_cliperase/transactions'
    subprocess.run([sys.executable, str(ROOT / 'tools/plot_mission_map_visible_clusters.py'),
                    '--transactions', str(transactions), '--output', str(output)],
                   cwd=ROOT, env=env, check=True)
    map_data = bars.read(output / 'mission_map_1x2.json')
    assert map_data['uav_count'] == 10 and map_data['mission_stops'] == 5
    assert map_data['formation_positions_epsg3857'] == map_data['display_positions_epsg3857']
    assert len(map_data['route_arrows']) == 5
    assert [(r['from_visit'], r['to_visit']) for r in map_data['route_arrows']] == [(1,2),(2,3),(3,4),(4,5),(5,1)]
    assert len(map_data['per_uav_paths_epsg3857']) == 10
    for path in map_data['per_uav_paths_epsg3857']:
        assert len(path) == 6 and path[0] == path[-1]
    for row in map_data['snapshot_records']:
        source = ROOT / row['source']
        assert source.parent == transactions
        assert bars.sha(source) == row['source_sha256']
    for name, digest in {**registered, **inputs}.items():
        assert bars.sha(ROOT / name) == digest, name
    notes = """Paper figures — 2026-10-26

All PNG/PDF/SVG figures in this folder use completed v4 event-mission results.
Summary bars: 7.4 × 3.735 inch canvas; 10% lower than the preceding version.
Total energy uses 1800–2300 kJ with explicit axis breaks; the full-scale file starts at zero.

loss_convergence_1x2 / mr_convergence_1x2: actual saved retained-set evaluations
from FL round 5 through 200. The first 20 rounds are common initialization.
FU-only evaluations may share an FL-round x coordinate and are not removed.
Markers distinguish overlapping methods; values are not shifted or smoothed.
This single-seed trajectory is not proof of asymptotic convergence.
convergence_samples.csv provides every plotted sample and its source file.

mission_map_1x2: larger 10.4 × 5.35 inch map. Blue enclosures are visual
groupings of five snapshots of the same ten UAVs, not radio coverage.
Five centroid arrows show visits 1–2–3–4–5 and return to 1. Faint paths preserve
individual UAV identity. Orange circles are synthetic IoT demand; stars are
recorded optimized positions. Red stars are actual heads from the earliest
positive-duration Proposed CapERA transmission in each region of the v4 run.
Head selections are checked against the completion-driven mission timeline.
Satellite and map data attribution and provenance are in mission_map_1x2.json.

Reproduce: MPLCONFIGDIR=/tmp/ova_mpl_style .venv/bin/python tools/build_paper_figures_20261026.py
"""
    (output / 'README.txt').write_text(notes)
    source_paths = [Path(__file__), ROOT / 'tools/plot_event_mission_reference_style.py',
                    ROOT / 'tools/plot_mission_map_visible_clusters.py',
                    ROOT / 'tools/plot_mission_map_cluster.py', ROOT / 'tools/plot_mission_map_reference.py']
    bars.save_json(output / 'bundle_manifest.json', dict(
        campaign=str(campaign.relative_to(ROOT)), protocol_sha256=bars.sha(protocol_path),
        registered_sources_unchanged=True,
        sources={str(p.relative_to(ROOT)): bars.sha(p) for p in source_paths},
        bar_height_reduction_percent=10, map_figure_size_inches=[10.4,5.35],
        curves_final_values_verified=True, actual_head_timing_verified=True,
        curve_evaluation_counts=counts,
        files={p.name: bars.sha(p) for p in sorted(output.iterdir())
               if p.is_file() and p.name != 'bundle_manifest.json'}))
    print(json.dumps(dict(output=str(output), chart_count=9, series=counts,
                          map_heads=[r['head_uav_zero_based'] for r in map_data['snapshot_records']],
                          checks='passed'), indent=2))


if __name__ == '__main__':
    main()
