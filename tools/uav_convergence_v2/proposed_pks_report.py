"""Promote the explicitly selected P-KS identity; plot actual costs and learning.

The earlier adaptive Proposed and P-KB25 are not substituted into this bundle.
Historical experiments remain intact. New sensitivity runs use the frozen P-KS
training implementation and the same conditions as the three paper transfers.
"""
import html
import math
from pathlib import Path

import numpy as np
import learning_quality_report as export
import learning_time_report as continuation
import learning_selection_fix_v2 as selection

ROOT = export.ROOT
DEST = ROOT / 'final_final_charts/proposed_pks'
prior = export.prior
plots = export.plots
read, write, sha = export.read, export.write, export.sha
DATASETS = ['CapERA', 'AirScape']
METHODS = {'gate_select16': 'Proposed', 'iscc': 'ISCC-P2',
           'afl': 'AFL-OTA', 'salun': 'F-SalUn'}
TRANSFER_IDS = {'iscc': 'iscc_p2_gaft', 'afl': 'uav_afl_ota_gaft', 'salun': 'f_salun'}
AXES = {key: dict(plots.AXES[key]) for key in ['downlink', 'fl_rounds', 'packet_loss']}
PALETTE = [('#4C72B0', 'o', '-'), ('#EFAA54', 's', '--'),
           ('#66B366', 'v', '-.'), ('#C65365', '^', ':')]


def source(case, dataset, method):
    if method == 'gate_select16':
        return (selection.OUT / method / 'cases' / case /
                (dataset.lower() + '_seed0') / export.quality.PROPOSED)
    return export.source(case, dataset, TRANSFER_IDS[method])


def case_id(axis, value):
    return 'baseline' if value == AXES[axis]['base'] else axis + '_' + str(value).replace('.', 'p')


def cases():
    return ['baseline'] + [case_id(a, v) for a, spec in AXES.items()
                           for v in spec['values'] if v != spec['base']]


def setup():
    export.setup(DEST, METHODS, source)
    for key, index in [('COLORS', 0), ('MARKERS', 1), ('STYLES', 2)]:
        setattr(prior.chart, key, [p[index] for p in PALETTE])
        setattr(prior.chart.style, key, [p[index] for p in PALETTE])
    plots.source = source


def collect():
    rows, checks = [], []
    cache = {}

    def digest(path):
        key = str(path.resolve())
        if key not in cache:
            cache[key] = sha(path)
        return cache[key]

    for case in cases():
        for ds in DATASETS:
            anchor = source(case, ds, 'gate_select16')
            for method in METHODS:
                arm = source(case, ds, method)
                assert read(arm / 'outcome.json')['status'] == 'completed', str(arm)
                for name in ['plan.json', 'requests.json', 'initial20/final.pt',
                             'feature_cache/features.pt', 'reference200/final.pt']:
                    assert digest(arm.parent / name) == digest(anchor.parent / name), (case, ds, method, name)
                physical = read(arm.parents[1] / 'settings.json')['energy_model']
                assert physical == read(anchor.parents[1] / 'settings.json')['energy_model']
                row = plots.ledger(case, ds, method)
                tx = [prior.load(f) for f in sorted((arm / 'transactions').glob('*.json'))]
                row['uplink_s'] = sum(t['radio']['uplink_s'] for t in tx)
                row['downlink_s'] = sum(t['timing']['downlink_s'] for t in tx)
                row['downlink_fraction'] = row['downlink_s'] / row['delay_s']
                row['hover_energy_fraction'] = row['hover_j'] / row['total_energy_j']
                row['learning_identity'] = 'P-KS' if method == 'gate_select16' else METHODS[method]
                expected = int(case.split('_')[-1]) if case.startswith('fl_rounds_') else 200
                assert row['FL_rounds'] == expected and row['FU_cycles'] == 225
                assert math.isclose(row['total_energy_j'], row['compute_j'] + row['communication_j'] + row['hover_j'], rel_tol=1e-12)
                selection_count = 0
                if method == 'gate_select16':
                    requests = prior.load(arm / 'request_records.json')
                    plan = prior.load(arm.parent / 'plan.json')
                    for f in (arm / 'work_attempts').glob('*.json'):
                        w = prior.load(f)
                        deleted = {i for r in requests if r['arrival_s'] <= w['mission_time_s'] + 1e-9 for i in r['images']}
                        for record in w['compute']['FL_sample_selection']:
                            pool = set(plan['partitions'][record['user']]) - deleted
                            batch = record['selected']
                            assert len(batch) == len(set(batch)) == 16 and set(batch) <= pool
                            selection_count += 1
                    assert selection_count == 10 * (expected - 20)
                rows.append(row)
                checks.append(dict(case=case, dataset=ds, method=method,
                                   shared_inputs_and_physics=True, costs_recomputed=True,
                                   completed_FL_rounds=expected, FU_cycles=225,
                                   causal_selected_batches=selection_count,
                                   source=str(arm.relative_to(ROOT))))
    return rows, checks


def round_curves():
    paths = {(ds, m): prior.trajectories(ds, m) for ds in DATASETS for m in METHODS}
    for number, (metric, label, scale) in enumerate([
            ('loss', 'Validation loss', 1.), ('mR', 'Retained mR (%)', 100.),
            ('KL', 'Forget-set KL', 1.)], 1):
        name = f'{number:02d}_round_{metric.lower()}_1x2'
        fig, axes = prior.chart.style.canvas(label)
        panels = []
        for ax, ds in zip(axes, DATASETS):
            values = []
            for i, (m, title) in enumerate(METHODS.items()):
                warm, points, _ = paths[(ds, m)]
                x = [p['completed_FL_rounds'] for p in points]
                y = [p['evaluation']['forget']['KL_retrain_to_candidate'] if metric == 'KL'
                     else scale * p['evaluation']['retained'][metric] for p in points]
                if metric != 'KL':
                    x = [k for k, _ in warm] + x
                    y = [scale * v[metric] for _, v in warm] + y
                line = prior.chart.style.curve(ax, x, y, i, title)
                # Match actual FL rounds across methods, rather than record indices.
                # Several FU evaluations may share a round: mark the last one,
                # while retaining every original point in the connecting line.
                marker_rounds = list(range(25 if metric == 'KL' else 0, 201, 25))
                last_at_round = {round_number: index for index, round_number in enumerate(x)}
                marker_indices = [last_at_round[round_number] for round_number in marker_rounds]
                line.set_markevery(marker_indices)
                values += y
                prior.SERIES.append(dict(figure=name, dataset=ds, method=m, metric=metric,
                                         domain='round', x=x, y=y, no_smoothing=True,
                                         marker_rounds=marker_rounds, marker_indices=marker_indices,
                                         duplicate_round_marker='last observed evaluation',
                                         reference='common_final200' if metric == 'KL' else None))
            prior.chart.regular_axis(ax, 'x', [0, 50, 100, 150, 200], 0)
            ax.set_xlabel('FL round')
            panels.append(values)
        prior.chart.finish_axes(axes, panels, decimals=2 if metric == 'loss' else 1 if metric == 'mR' else None)
        prior.save(fig, axes, name, label + ' versus FL round (fixed 200-round workload)')


def time_curves():
    paths, checks, budget = {}, [], []
    for ds in DATASETS:
        for m in METHODS:
            paths[(ds, m)], progress, check = continuation.trajectory(ds, m)
            checks.append(check)
            p = next(p for p in reversed(paths[(ds, m)]) if p['active_s'] <= 400)
            budget.append(dict(dataset=ds, method=m, label=METHODS[m],
                               checkpoint_s=p['active_s'], FL_round=p['FL_round'],
                               mR_percent=100 * p['evaluation']['retained']['mR'],
                               loss=p['evaluation']['retained']['loss']))
    for number, (metric, label, scale) in enumerate([
            ('loss', 'Validation loss', 1.), ('mR', 'Retained mR (%)', 100.)], 4):
        name = f'{number:02d}_time_{metric.lower()}_1x2'
        fig, axes = prior.chart.style.canvas(label)
        panels = []
        for ax, ds in zip(axes, DATASETS):
            values = []
            for i, (m, title) in enumerate(METHODS.items()):
                points = [p for p in paths[(ds, m)] if p['active_s'] <= 400]
                x = [p['active_s'] for p in points]
                y = [scale * p['evaluation']['retained'][metric] for p in points]
                dx, dy = x + [400.], y + [y[-1]]
                line = prior.chart.style.curve(ax, dx, dy, i, title)
                line.set_drawstyle('steps-post')
                values += y
                prior.SERIES.append(dict(figure=name, dataset=ds, method=m, metric=metric,
                                         domain='time', x=x, y=y, display_x=dx, display_y=dy,
                                         drawstyle='steps-post', actual_continued_training=True))
            prior.chart.regular_axis(ax, 'x', [0, 100, 200, 300, 400], 0)
            ax.set_xlabel('Time (s)')
            panels.append(values)
        prior.chart.finish_axes(axes, panels, decimals=2 if metric == 'loss' else 1)
        prior.save(fig, axes, name, label + ' under a shared 400-second budget')
    write(DEST / 'data/budget400.json', budget)
    prior.write_csv(DEST / 'data/budget400.csv', budget)
    return checks


def sweep_curves(rows):
    original_axes, original_methods = plots.AXES, plots.study.METHODS
    plots.AXES, plots.study.METHODS = AXES, METHODS
    try:
        for i, a in enumerate(AXES):
            plots.draw(rows, a, 'total_energy_j', 'Total energy (kJ)', .001, f'{6 + 2*i:02d}_{a}_energy_1x2')
            plots.draw(rows, a, 'delay_s', 'Delay (s)', 1., f'{7 + 2*i:02d}_{a}_delay_1x2')
            plots.draw(rows, a, 'mission_delay_s', 'Mission delay (s)', 1., f'{12 + i:02d}_{a}_mission_delay_1x2')
        return plots.response_summary(rows)
    finally:
        plots.AXES, plots.study.METHODS = original_axes, original_methods


def docs(rows, responses):
    text = ['# Proposed = P-KS: learning and resource comparisons', '',
            'The user-selected P-KS is now the only method labeled **Proposed** in this bundle. It is the unchanged proxy-KL-gated FU + half-exploration/half-loss-priority FL16 method. The previous adaptive-FU/uniform-FL Proposed is removed from these comparisons. P-KB25 is a separate experiment and is not substituted for P-KS. Historical bundles retain their original identities.', '',
            'Methods: Proposed, ISCC-P2, AFL-OTA and F-SalUn. The [paper-transfer definitions](../four_methods/README.md) remain unchanged. Main line color/marker for Proposed is blue/circle. Every chart is CapERA left and AirScape right, with uniform endpoint ticks, matching panel precision, and no value labels or extra captions. Round-curve markers share the same 25-round grid across all methods; the last observed evaluation is marked when FU produces multiple observations at one FL round. All original line values remain unchanged.', '',
            '## Figures', '',
            '- 01–03: validation loss, retained mR and forget-set KL versus FL round, through the common 200-round workload. FU updates occur during this workload; the loss is retained retrieval validation CE after the actual FL/FU updates, not a sum of incompatible local FL and FU objectives.',
            '- 04–05: loss and mR versus active time, with actual retained-only FL continuation after all 45 requests finish. All methods receive the same continuation learning rate and 400-second budget. Display the last evaluated completed checkpoint; no invented additional progress.',
            '- 06–11: total energy and active FL/FU delay versus downlink rate, total FL-round budget and packet loss.',
            '- 12–14: mission delay for those same three axes, kept separate from active service delay.', '',
            '## Why these axes change costs', '',
            'In this cached-adapter scenario, full-model broadcasts and active hovering dominate cost. Increasing uplink bandwidth changes only one part of service time. At a fixed 400-second hover budget, total hover energy is almost fixed by construction; resource comparisons therefore use completed prescribed work, not equal elapsed-time budgets.', '',
            '| Axis | Values | Intervention |', '|---|---|---|',
            '| Downlink rate | 0.25, 0.5, 1, 2, 4 Mbit/s | Same full update payload and work; change the model broadcast rate. |',
            '| FL-round budget | 100, 125, 150, 175, 200 | Same 20 warmup rounds and all 45 requests/225 FU cycles; change subsequent FL work. Original cosine schedule is truncated, not retuned. Release ordinals scale within each site. |',
            '| Downlink packet loss | 0, 10, 20, 30, 40% | Repeat failed whole-model broadcasts until success; charge actual attempts. This is a declared erasure model, not a measured link. |', '',
            'Every condition uses identical scenario inputs, physical coefficients and prescribed workload across the four methods. All 24 nonbaseline P-KS dataset/condition arms are fresh executions of the frozen registered algorithm. The matching 78 paper-transfer arms and two original P-KS baseline arms are reused unchanged: 104 unique arms total. Neither flat curves nor unfavorable results are rescaled or removed. Energy is active communication + executed adapter computation + hovering; travel, IoT collection and preflight are excluded. Mission delay separately includes travel, collection and return. The fixed backbone feature cache excludes full-VLM encoding costs.', '',
            '## KL interpretation and limits', '',
            'The round-wise KL compares bidirectional retrieval output distributions on the forget set against the common retained-only retraining model at round 200, using D_KL(reference || candidate). It is not the training proxy KL. Early-round distances also reflect the different training horizons, so they are descriptive and do not certify current-request unlearning. Shorter FL-budget cases use the same fixed reference; there is no newly matched shorter reference claim.', '',
            'P-KS has the previously reported larger common-reference KL; renaming it does not improve it. The much smaller P-KB25 values from the [separate KL repair study](../pks_kl_repair/README.md) do not belong to this Proposed. Both common and policy-matched reference diagnostics remain available there under their historical method names.', '',
            'These are single-seed development results with the original validation subsets. The known-cohort training proxy assumption remains; the reporting retrain is never a training teacher. Completed requests mean all prescribed FU steps finished, not proof of deletion. Fewer FL rounds also change quality. All final utility/KL and physical bound exceedances remain in the aggregate metrics.', '',
            '## Baseline cost components', '',
            '| Dataset | Method | Energy (kJ) | Active delay (s) | Broadcast share of delay | Hover share of energy |',
            '|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['case'] == 'baseline':
            text.append(f"| {r['dataset']} | {r['label']} | {r['total_energy_j']/1000:.3f} | {r['delay_s']:.3f} | {100*r['downlink_fraction']:.2f}% | {100*r['hover_energy_fraction']:.2f}% |")
    text += ['', '## Measured endpoint changes for Proposed', '',
             '| Axis | Dataset | Energy change | Active-delay change | Energy shape |',
             '|---|---|---:|---:|---|']
    for a in AXES:
        for ds in DATASETS:
            e = next(r for r in responses if r['axis'] == a and r['dataset'] == ds and r['method'] == 'gate_select16' and r['metric'] == 'total_energy_j')
            d = next(r for r in responses if r['axis'] == a and r['dataset'] == ds and r['method'] == 'gate_select16' and r['metric'] == 'delay_s')
            text.append(f"| {a} | {ds} | {e['endpoint_change_percent']:+.2f}% | {d['endpoint_change_percent']:+.2f}% | {e['direction']} |")
    names = ' '.join(c for c in cases() if c != 'baseline')
    text += ['', '## Reproduction', '',
             'With existing local datasets/caches and the baseline/transfer experiments, run:', '',
             '```sh', '.venv/bin/python tools/uav_convergence_v2/learning_quality_runner_v2.py run --corrected --variants gate_select16 --cases ' + names,
             '.venv/bin/python tools/uav_convergence_v2/proposed_pks_report.py',
             '.venv/bin/python tools/uav_convergence_v2/proposed_pks_check.py', '```', '',
             'The frozen runner preserves earlier experiments. Only code, charts, aggregate metrics and provenance are published. Checkpoints, raw samples and execution logs remain local.', '']
    (DEST / 'README.md').write_text('\n'.join(text))


def landing_page():
    cards = []
    for figure in prior.FIGURES:
        stem = 'proposed_pks/' + figure['name']
        cards.append('<section><h2>' + html.escape(figure['title']) + '</h2><a href="' +
                     stem + '.pdf"><img loading="lazy" src="' + stem + '.png"></a></section>')
    page = ('<!doctype html><meta charset="utf-8"><title>Proposed P-KS comparisons</title>'
            '<style>body{font:16px system-ui;max-width:1150px;margin:35px auto;padding:0 20px;'
            'color:#202530}img{width:100%;height:auto}h2{font-size:20px}section{margin:45px 0}</style>'
            '<h1>UAV/VLM comparisons</h1><p><strong>Proposed = P-KS.</strong> '
            'CapERA · AirScape · ISCC-P2 · AFL-OTA · F-SalUn.</p>'
            '<p><a href="proposed_pks/README.md">Methods, experiment conditions and results</a> · '
            '<a href="proposed_pks/data/metrics.csv">Metrics CSV</a></p>' + ''.join(cards) +
            '<hr><p>Historical bundles retain their original method names: '
            '<a href="four_methods/index.html">Earlier four-method comparison</a> · '
            '<a href="pks_kl_repair/index.html">P-KS KL audit and P-KB25 study</a> · '
            '<a href="learning_quality/index.html">Learning diagnostics</a> · '
            '<a href="same_time/index.html">Earlier continuation study</a> · '
            '<a href="expanded_sweeps/index.html">Earlier sweeps</a></p>\n')
    (DEST.parent / 'index.html').write_text(page)


def main():
    selection.register()
    setup()
    rows, checks = collect()
    assert len(rows) == 104
    round_curves()
    time_checks = time_curves()
    responses = sweep_curves(rows)
    docs(rows, responses)
    write(DEST / 'data/responses.json', responses)
    prior.write_csv(DEST / 'data/responses.csv', [{k: v for k, v in r.items() if k not in ['x', 'values']} for r in responses])
    write(DEST / 'data/audit.json', dict(passed=True, arms=104, new_PKS_arms=24,
                                      shared_condition_checks=checks, continued_time_checks=time_checks))
    write(DEST / 'data/method_identity.json', dict(methods=METHODS,
          proposed_source_variant='gate_select16', historical_label='P-KS',
          original_adaptive_proposed_included=False, PKB25_substituted=False,
          training_protocol_sha256=sha(selection.OUT / 'protocol.json'),
          source_sha256=selection.register()['source_sha256']))
    prior.SOURCES[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    prior.FIGURES.sort(key=lambda figure: figure['name'])
    export.finish(DEST, rows, 'Proposed = P-KS; round/time learning and three resource sensitivities.',
                  dict(compared_arms=104, new_PKS_arms=24, sensitivity_axes=AXES))
    landing_page()
    print(dict(figures=len(prior.FIGURES), arms=len(rows), destination=str(DEST)), flush=True)


if __name__ == '__main__':
    main()
