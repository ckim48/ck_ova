"""Observed FL/FU co-aggregation, matched serial costs and useful saved time."""
from pathlib import Path
import math
import numpy as np
from matplotlib.patches import Patch

import pks_parallel_ablation as study
import proposed_pks_report as current

ROOT = study.ROOT
DEST = ROOT / 'final_final_charts/parallel_advantage'
read, write, sha = study.read, study.write, study.sha
export, prior = current.export, current.prior
PAIR = {'proposed': 'Proposed', 'serial': 'S-PKS'}
MAIN = {'proposed': 'Proposed', 'iscc': 'ISCC-P2', 'afl': 'AFL-OTA', 'salun': 'F-SalUn'}
ALL = {**MAIN, 'serial': 'S-PKS'}
DATASETS = current.DATASETS


def source(case, ds, method):
    if method in PAIR:
        return study.source('parallel' if method == 'proposed' else 'serial', ds)
    return current.source(case, ds, 'gate_select16' if method == 'gate_select16' else method)


def styles(methods):
    prior.METHODS = methods
    palette = {'proposed': ('#4C72B0', 'o', '-'), 'serial': ('#92703B', 'd', '--'),
               'iscc': ('#EFAA54', 's', '--'), 'afl': ('#66B366', 'v', '-.'),
               'salun': ('#C65365', '^', ':')}
    for key, index in [('COLORS', 0), ('MARKERS', 1), ('STYLES', 2)]:
        values = [palette[m][index] for m in methods]
        setattr(prior.chart, key, values)
        setattr(prior.chart.style, key, values)


def schedule(arm, include_window=True):
    timeline = prior.load(arm / 'mission_timeline.json')
    tx = [prior.load(p) for p in sorted((arm / 'transactions').glob('*.json'))]
    requests = prior.load(arm / 'request_records.json')
    assert len(requests) == 45 and sum(r['steps'] for r in requests) == 225
    assert sum(t['kind'] in ['fl', 'joint'] for t in tx) == 180
    counts = {kind: sum(t['kind'] == kind for t in tx) for kind in ['fl', 'fu', 'joint']}
    counts['broadcasts'] = sum(int(t['radio']['uplink_s'] > 0 or t['radio'].get('broadcast_required', False)) for t in tx)
    counts['joint_FU_steps'] = sum(len(t['FU_request_ids']) for t in tx if t['kind'] == 'joint')
    counts['total_FU_steps'] = sum(len(t['FU_request_ids']) for t in tx)
    counts['active_delay_s'] = sum(s['service_duration_s'] for s in timeline['sites'])
    counts['queue_wait_s'] = sum(w['duration_s'] for w in timeline['waits'])
    if not include_window:
        return counts, None
    # First chronological release, selected by event order, not quality or benefit.
    arrival = min(r['arrival_s'] for r in requests)
    first = [r for r in requests if math.isclose(r['arrival_s'], arrival, abs_tol=1e-10)]
    before = [t for t in tx if t['end_s'] <= arrival + 1e-9][-1]['completed_FL_rounds']
    required_end = max(r['completed_s'] for r in first)
    end = next(t['end_s'] for t in tx if t['end_s'] >= required_end - 1e-9 and t['completed_FL_rounds'] >= before + 5)
    events = [t for t in tx if t['start_s'] >= arrival - 1e-9 and t['end_s'] <= end + 1e-9]
    assert len(first) == 5 and sum(t['kind'] in ['fl', 'joint'] for t in events) == 5
    assert sum(len(t['FU_request_ids']) for t in events) == 25
    window = dict(duration_s=end - arrival, starting_FL_round=before,
                  extra_FL_rounds=5, FU_requests=5, FU_steps=25,
                  events=[dict(start_s=t['start_s'] - arrival, end_s=t['end_s'] - arrival,
                               FL=t['kind'] in ['fl', 'joint'], FU=bool(t['FU_request_ids'])) for t in events])
    assert math.isclose(sum(e['end_s'] - e['start_s'] for e in window['events']), window['duration_s'], abs_tol=1e-8)
    return counts, window


def timeline_chart(windows):
    name = '01_parallel_service_timeline_1x2'
    fig, axes = prior.chart.style.canvas('')
    colors = {'FL': '#4C72B0', 'FU': '#EFAA54'}
    panels = []
    for ax, ds in zip(axes, DATASETS):
        for method, center in [('serial', .5), ('proposed', 2.5)]:
            window = windows[(ds, method)]
            for event in window['events']:
                for kind, offset in [('FL', .23), ('FU', -.23)]:
                    if event[kind]:
                        ax.broken_barh([(event['start_s'], event['end_s'] - event['start_s'])],
                                       (center + offset - .19, .38), facecolors=colors[kind],
                                       edgecolors='white', linewidth=.7,
                                       hatch='//' if kind == 'FL' else '\\\\')
            prior.SERIES.append(dict(figure=name, dataset=ds, method=method,
                                     kind='service_window', **window))
        prior.chart.regular_axis(ax, 'x', [0, 2, 4, 6, 8], 0)
        ax.set_xlabel('Time since FU release (s)')
        ax.set_ylim(-.4, 3.4)
        ax.set_yticks([.5, 2.5], ['S-PKS', 'Proposed'])
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(False, axis='y')
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, .015, f'({chr(97 + DATASETS.index(ds))}) {ds}',
                 ha='center', va='bottom', fontsize=14)
        assert not ax.texts
        panels.append(dict(dataset=ds, x=dict(ticks=[0, 2, 4, 6, 8], limits=[0., 8.],
                           labels=['0', '2', '4', '6', '8']),
                           y=dict(kind='categorical', ticks=[.5, 2.5], limits=[-.4, 3.4], labels=['S-PKS', 'Proposed'])))
    fig.legend(handles=[Patch(facecolor=colors[k], edgecolor='white', hatch='//' if k == 'FL' else '\\\\', label=k + ' service') for k in ['FL', 'FU']],
               ncol=2, loc='upper center', bbox_to_anchor=(.5, 1.), fontsize=11)
    files = []
    for ext in ['png', 'pdf', 'svg']:
        path = DEST / (name + '.' + ext)
        fig.savefig(path, dpi=350, bbox_inches='tight')
        files.append(dict(file=path.name, sha256=sha(path)))
    prior.chart.plt.close(fig)
    prior.FIGURES.append(dict(name=name, title='Same first release plus five FL updates: overlapping service versus immediate serial service',
                             axes=panels, files=files))


def cost_bars(rows):
    styles(PAIR)
    for number, (metric, label, scale) in enumerate([
            ('delay_s', 'Delay (s)', 1.), ('total_energy_j', 'Total energy (kJ)', .001)], 2):
        name = f'{number:02d}_matched_{"delay" if metric == "delay_s" else "energy"}_1x2'
        fig, axes = prior.chart.style.canvas(label)
        panels = []
        for ax, ds in zip(axes, DATASETS):
            values = []
            for index, (m, title) in enumerate(PAIR.items()):
                row = next(r for r in rows if r['dataset'] == ds and r['method'] == m)
                value = row[metric] * scale
                ax.bar(index + 1, value, width=.65, color=prior.chart.COLORS[index],
                       edgecolor='black', linewidth=.7, hatch='//' if index == 0 else '..', label=title)
                values.append(value)
                prior.SERIES.append(dict(figure=name, dataset=ds, method=m, metric=metric,
                                         x=[index + 1], y=[value], scale=scale, bar=True))
            prior.chart.regular_axis(ax, 'x', [0, 1, 2, 3], 0)
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            panels.append([0.] + values)
        prior.chart.finish_axes(axes, panels, decimals=0)
        # Cost bars have a true zero baseline.
        for ax in axes:
            ticks = [v for v in ax.get_yticks() if v >= 0]
            assert ticks[0] == 0
            prior.chart.regular_axis(ax, 'y', ticks, 0)
        prior.save(fig, axes, name, label + ': identical P-KS learning rules, parallel versus immediate serial')


def progress_curves():
    styles(MAIN)
    paths, checks, budgets = {}, [], []
    for ds in DATASETS:
        for method in MAIN:
            original = 'gate_select16' if method == 'proposed' else method
            _, progress, check = current.continuation.trajectory(ds, original)
            paths[(ds, method)] = [p for p in progress if p['active_s'] <= 400]
            check['method'] = method
            checks.append(check)
            budgets.append(dict(dataset=ds, method=method, **paths[(ds, method)][-1]))
    for number, (metric, label, ticks) in enumerate([
            ('FL_round', 'Completed FL rounds', [0, 100, 200, 300, 400, 500, 600, 700]),
            ('FU_cycles', 'Completed FU steps', [0, 75, 150, 225])], 4):
        name = f'{number:02d}_time_{"fl" if metric == "FL_round" else "fu"}_progress_1x2'
        fig, axes = prior.chart.style.canvas(label)
        for ax, ds in zip(axes, DATASETS):
            for index, (method, title) in enumerate(MAIN.items()):
                p = paths[(ds, method)]
                x, y = [v['active_s'] for v in p], [v[metric] for v in p]
                dx, dy = x + [400.], y + [y[-1]]
                line = prior.chart.style.curve(ax, dx, dy, index, title)
                line.set_drawstyle('steps-post')
                prior.SERIES.append(dict(figure=name, dataset=ds, method=method, metric=metric,
                                         x=x, y=y, display_x=dx, display_y=dy, drawstyle='steps-post'))
            prior.chart.regular_axis(ax, 'x', [0, 100, 200, 300, 400], 0)
            prior.chart.regular_axis(ax, 'y', ticks, 0)
            ax.set_xlabel('Time (s)')
        prior.save(fig, axes, name, label + ' with actual use of the shared 400-second budget')
    return checks, budgets


def main():
    protocol = study.register()
    control = read(study.OUT / 'verification.json')
    assert control['passed'] and control['original_PKS_reproduced_exactly']
    export.setup(DEST, ALL, source)
    current.plots.source = source
    rows, schedules, windows = [], [], {}
    for ds in DATASETS:
        for m in PAIR:
            row = current.plots.ledger('baseline', ds, m)
            counts, window = schedule(source('baseline', ds, m))
            row.update(counts)
            row['first_release_plus_five_FL_s'] = window['duration_s']
            rows.append(row)
            windows[(ds, m)] = window
        for m in ['iscc', 'afl', 'salun']:
            counts, _ = schedule(source('baseline', ds, m), include_window=False)
            schedules.append(dict(dataset=ds, method=m, **counts))
    # Exact prefix release timing and same FL/FU work; serial has no periodic wait.
    for ds in DATASETS:
        assert windows[(ds, 'proposed')]['starting_FL_round'] == windows[(ds, 'serial')]['starting_FL_round'] == 25
    timeline_chart(windows)
    cost_bars(rows)
    checks, budgets = progress_curves()
    changes = []
    for ds in DATASETS:
        a = next(r for r in rows if r['dataset'] == ds and r['method'] == 'proposed')
        b = next(r for r in rows if r['dataset'] == ds and r['method'] == 'serial')
        changes.append(dict(dataset=ds, delay_reduction_percent=100 * (1 - a['delay_s']/b['delay_s']),
                            energy_reduction_percent=100 * (1 - a['total_energy_j']/b['total_energy_j']),
                            window_reduction_percent=100 * (1 - a['first_release_plus_five_FL_s']/b['first_release_plus_five_FL_s']),
                            broadcast_reduction_percent=100 * (1 - a['broadcasts']/b['broadcasts'])))
    write(DEST / 'data/protocol.json', protocol)
    write(DEST / 'data/audit.json', dict(passed=True, matched_control=control, continuation_checks=checks))
    write(DEST / 'data/improvements.json', changes)
    write(DEST / 'data/literature_scheduling.json', schedules)
    write(DEST / 'data/budget400.json', budgets)
    prior.write_csv(DEST / 'data/budget400.csv', budgets)
    docs(rows, changes)
    prior.METHODS = ALL
    for p in [Path(__file__), Path(study.__file__)]:
        prior.SOURCES[str(p.relative_to(ROOT))] = sha(p)
    export.finish(DEST, rows, 'Parallel FL/FU co-aggregation: matched serial ablation and measured completed work.',
                  dict(fresh_control_arms=4, proposed_identity='P-KS'))
    link_galleries()
    print(dict(figures=len(prior.FIGURES), improvements=changes), flush=True)


def docs(rows, changes):
    text = ['# Parallel FL/FU advantage: Proposed (P-KS)', '',
            'The parallelism is **FL/FU co-aggregation and model application in the same service transaction**. Local FL and FU gradient computation is charged as executed work on each UAV; the simulator does not assume two gradients execute simultaneously on one processor.', '',
            '## Matched scheduling ablation', '',
            '**S-PKS** uses the identical P-KS objectives, training-only sample selection, initialization, data, requests, optimized AirComp/head-selection algorithm, learning-rate rules and physical coefficients. Only scheduling changes: pending FU is served immediately in separate transactions (`parallel=False, period=0`), then FL resumes. Multiple FU clients still aggregate via AirComp. No artificial periodic wait or OFDMA substitution is introduced. Scheduling changes model states and optimized physical decisions as consequences. Both methods finish 200 total FL rounds and 225 request-local FU steps for 45 requests.', '',
            'Four fresh runs cover the two modes and two datasets. The new parallel controls reproduce the original P-KS final parameters, evaluation, transaction order and physical timings/energy exactly. All serial outcomes remain reported. Gains are for the same prescribed work, not proof of equal final quality or certified unlearning.', '',
            'The mean latency of an individual FU request is approximately 3.384 seconds in both modes. Immediate serial service also begins FU promptly; its disadvantage here is pausing FL during that service. Thus the approximately 50% first-window reduction refers to completing FU and five FL updates together, not halving FU-only latency. The full-workload reduction is approximately 20.1%.', '',
            '## Figures', '',
            '- 01: the first chronological FU release, containing five requests × five FU steps, plus five additional FL updates. Both modes start after FL round 25. Colored intervals are actual service transactions including compute, uplink and broadcast. Coincident FL/FU lanes mean the updates are jointly served and applied; they do not claim simultaneous CPU execution. Every mode completes exactly five FL updates and 25 request-local FU steps in this window. This local window is about half as long under parallel service; it is not the total-mission speedup.',
            '- 02–03: actual full-workload delay and energy for Proposed versus S-PKS.',
            '- 04–05: completed FL rounds and prescribed FU steps versus active time for the four main methods. After all initial work finishes, each uses remaining time for retained-only FL under the already declared common continuation learning rate. The FL axis starts with 20 shared warmup rounds already complete. No new FU requests arrive during continuation. Completed steps describe service, not deletion quality.', '',
            'In the baseline, Proposed uses 135 FL-only plus 45 joint transactions. S-PKS and the literature transfers use 180 FL-only plus 45 FU-only transactions. Actual broadcasts are 179 versus 224 because the terminal zero-learning-rate FL update does not send a model. Joint service covers all 225 request-local FU steps. A transaction serving five requests counts as five FU steps, not five broadcasts.', '',
            'The literature curves compare complete adapted methods with different learning and radio rules. Their entire gap cannot be assigned solely to parallelism. The matched S-PKS comparison isolates this scheduling change within the P-KS design. Arrival releases are tied to FL ordinals, not a common exogenous wall-clock process; this is not a stochastic-arrival capacity benchmark.', '',
            'Total energy is active communication + executed adapter computation + hover. Delay excludes flight, collection and preflight. Fixed feature caching and the known-cohort training-proxy assumption remain. Single development seed; no statistical superiority or deletion guarantee. The reported KL remains D_KL(retrain || candidate) against the common retained-only round-200 reference.', '',
            '| Dataset | Method | Delay (s) | Energy (kJ) | First-release window (s) | mR (%) | Loss | Mean KL | Worst KL |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['dataset']} | {r['label']} | {r['delay_s']:.3f} | {r['total_energy_j']/1000:.3f} | {r['first_release_plus_five_FL_s']:.3f} | {r['retained_mR_percent']:.3f} | {r['retained_loss']:.5f} | {r['forget_KL']:.6g} | {r['worst_request_KL']:.6g} |")
    text += ['', '| Dataset | Full-workload delay reduction | Energy reduction | First-release window reduction | Broadcast reduction |',
             '|---|---:|---:|---:|---:|']
    for r in changes:
        text.append(f"| {r['dataset']} | {r['delay_reduction_percent']:.2f}% | {r['energy_reduction_percent']:.2f}% | {r['window_reduction_percent']:.2f}% | {r['broadcast_reduction_percent']:.2f}% |")
    text += ['', '## Reproduction', '', '```sh',
             '.venv/bin/python tools/uav_convergence_v2/pks_parallel_ablation.py run',
             '.venv/bin/python tools/uav_convergence_v2/parallel_advantage_report.py',
             '.venv/bin/python tools/uav_convergence_v2/parallel_advantage_check.py', '```', '',
             'Requires the existing local data/cache and previous main/continuation runs. Raw inputs, parameters and execution traces remain local. Published outputs contain code, figures, aggregated plot values and provenance.', '']
    (DEST / 'README.md').write_text('\n'.join(text))


def link_galleries():
    for path, prefix in [(DEST.parent / 'index.html', 'parallel_advantage/'),
                         (DEST.parent / 'proposed_pks/index.html', '../parallel_advantage/')]:
        page = path.read_text()
        href = prefix + 'index.html'
        if 'href="' + href + '"' not in page:
            position = page.index('</h1>') + len('</h1>')
            page = (page[:position] + '<p><strong>Parallel FL/FU advantage:</strong> <a href="' + href +
                    '">Service timeline, matched serial costs and completed-work curves</a></p>' + page[position:])
            path.write_text(page)
    for path, prefix in [(DEST.parent / 'README.md', 'parallel_advantage/'),
                         (DEST.parent / 'proposed_pks/README.md', '../parallel_advantage/')]:
        page = path.read_text()
        line = '**Parallel advantage:** [Five figures and a matched immediate-serial P-KS experiment](' + prefix + 'README.md).'
        if line not in page:
            first, rest = page.split('\n', 1)
            path.write_text(first + '\n\n' + line + '\n' + rest)


if __name__ == '__main__':
    main()
