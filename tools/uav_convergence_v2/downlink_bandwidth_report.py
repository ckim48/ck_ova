"""Export audited bandwidth sweeps and matching unannotated hatched bars."""
import copy
import html
import math
from pathlib import Path

import numpy as np
import downlink_bandwidth_v1 as study
import proposed_pks_report as current

ROOT, read, write, sha = study.ROOT, study.read, study.write, study.sha
DEST = ROOT / 'final_final_charts/downlink_bandwidth'
prior, export, plots = current.prior, current.export, current.plots
METHODS, DATASETS = study.METHODS, study.MODEL_DATASETS
HATCHES = ['//', '\\\\', '..', 'xx']
PALETTE = current.PALETTE


def source(case, dataset, method):
    value = int(case.rsplit('_', 1)[-1])
    if value in study.FRESH_VALUES:
        return study.fresh_source(value, dataset, method)
    return current.source('downlink_' + str(value), dataset, method)


def normalized_physics(settings):
    p = copy.deepcopy(settings['energy_model'])
    p.pop('downlink_bandwidth_model', None)
    c = p['communication']
    for k in ['downlink_bandwidth_hz', 'downlink_spectral_efficiency_bps_hz', 'downlink_rate_model']:
        c.pop(k, None)
    c['downlink_rate_bps'] = 1e6
    return p


def collect():
    import torch
    rows, checks, controls = [], [], []
    hashes = {}
    def digest(path):
        key = str(path.resolve())
        if key not in hashes:
            hashes[key] = sha(path)
        prior.SOURCES[str(path.relative_to(ROOT))] = hashes[key]
        return hashes[key]
    for bw in study.VALUES:
        case = study.case_id(bw)
        for ds in DATASETS:
            anchor = current.source('baseline', ds, 'gate_select16')
            anchor_settings = prior.load(anchor.parents[1] / 'settings.json')
            for m in METHODS:
                arm = source(case, ds, m)
                settings = prior.load(arm.parents[1] / 'settings.json')
                assert settings['config'] == anchor_settings['config']
                assert normalized_physics(settings) == normalized_physics(anchor_settings)
                assert settings['network_cap_j'] == anchor_settings['network_cap_j']
                for name in ['plan.json', 'requests.json', 'initial20/final.pt', 'reference200/final.pt', 'feature_cache/features.pt']:
                    assert digest(arm.parent / name) == digest(anchor.parent / name), (bw, ds, m, name)
                for name in ['deployment.json', 'mission_spec.json']:
                    assert digest(arm.parents[1] / name) == digest(anchor.parents[1] / name), (bw, ds, m, name)
                p = settings['energy_model']['communication']
                rate = study.rate_bps(bw * 1e6)
                assert p['downlink_rate_bps'] == rate
                if bw in study.FRESH_VALUES:
                    assert p['downlink_bandwidth_hz'] == bw * 1e6
                    assert p['downlink_spectral_efficiency_bps_hz'] == study.EFFICIENCY
                tx = [prior.load(f) for f in sorted((arm / 'transactions').glob('*.json'))]
                broadcast_time = 0.
                for t in tx:
                    radio = t['radio']
                    count = radio.get('broadcast_attempts', int(radio['uplink_s'] > 0 or radio.get('broadcast_required', False)))
                    expected = count * (p['update_coordinates'] * p['broadcast_bits_per_coordinate'] + p['control_bits_per_update']) / rate
                    assert math.isclose(expected, t['timing']['downlink_s'], abs_tol=1e-10)
                    broadcast_time += expected
                row = plots.ledger(case, ds, m)
                assert row['FL_rounds'] == 200 and row['FU_cycles'] == 225
                row.update(bandwidth_mhz=bw, spectral_efficiency_bps_hz=study.EFFICIENCY,
                           downlink_rate_bps=rate, downlink_s=broadcast_time,
                           origin='fresh_bandwidth_execution' if bw in study.FRESH_VALUES else 'equivalent_rate_reuse',
                           learning_identity='P-KS' if m == 'gate_select16' else METHODS[m])
                rows.append(row)
                checks.append(dict(case=case, dataset=ds, method=m, origin=row['origin'],
                                   identical_scenario_inputs=True, derived_rate_verified=True,
                                   all_broadcast_durations_verified=True, costs_recomputed=True,
                                   full_FL_FU_workload_completed=True, source=str(arm.relative_to(ROOT))))
                if bw == 1:
                    old = current.source('baseline', ds, m)
                    theta = torch.load(arm / 'final.pt', map_location='cpu', weights_only=True)['theta']
                    before = torch.load(old / 'final.pt', map_location='cpu', weights_only=True)['theta']
                    assert torch.equal(theta, before), (ds, m, 'baseline model')
                    assert prior.load(arm / 'service_completion.json')['evaluation'] == prior.load(old / 'service_completion.json')['evaluation']
                    previous = [prior.load(f) for f in sorted((old / 'transactions').glob('*.json'))]
                    assert len(previous) == len(tx)
                    for a, b in zip(previous, tx):
                        for key in ['kind', 'FU_request_ids', 'completed_FL_rounds']:
                            assert a[key] == b[key], (ds, m, key)
                        for key in ['start_s', 'end_s']:
                            assert math.isclose(a[key], b[key], abs_tol=1e-9), (ds, m, key)
                        for key in ['energy_bound_j', 'uplink_s', 'head', 'conditional_rmse']:
                            assert a['radio'][key] == b['radio'][key], (ds, m, key)
                    controls.append(dict(dataset=ds, method=m, identical_model_and_quality=True,
                                         identical_transaction_order_and_radio=True, clock_reproduced=True))
    assert len(rows) == 40 and len(controls) == 8
    return rows, dict(passed=True, arms=40, fresh_arms=24, reused_arms=16,
                      shared_condition_checks=checks, baseline_controls=controls)


def setup():
    export.setup(DEST, METHODS, source)
    plots.source = source
    for key, index in [('COLORS', 0), ('MARKERS', 1), ('STYLES', 2)]:
        setattr(prior.chart, key, [v[index] for v in PALETTE])
        setattr(prior.chart.style, key, [v[index] for v in PALETTE])
    prior.chart.plt.rcParams['hatch.linewidth'] = .55


def sweep(rows):
    by = {(r['bandwidth_mhz'], r['dataset'], r['method']): r for r in rows}
    for number, metric, label, scale, kind in [
            (1, 'total_energy_j', 'Total energy (kJ)', .001, 'energy'),
            (2, 'delay_s', 'Delay (s)', 1., 'delay'),
            (3, 'mission_delay_s', 'Mission delay (s)', 1., 'mission_delay')]:
        name = f'{number:02d}_downlink_bandwidth_{kind}_1x2'
        fig, axes = prior.chart.style.canvas(label)
        panels = []
        for ax, ds in zip(axes, DATASETS):
            values = []
            for i, (m, title) in enumerate(METHODS.items()):
                y = [by[(v, ds, m)][metric] * scale for v in study.VALUES]
                prior.chart.style.curve(ax, study.VALUES, y, i, title).set_markevery(1)
                values += y
                prior.SERIES.append(dict(figure=name, dataset=ds, method=m, kind='line',
                                         metric=metric, scale=scale, x=study.VALUES, y=y))
            prior.chart.regular_axis(ax, 'x', study.VALUES, 0)
            ax.set_xlabel('Downlink bandwidth (MHz)')
            panels.append(values)
        prior.chart.finish_axes(axes, panels)
        prior.save(fig, axes, name, label + ' versus downlink bandwidth')


def bars(rows):
    for number, metric, label, scale, kind in [
            (4, 'total_energy_j', 'Total energy (kJ)', .001, 'energy'),
            (5, 'delay_s', 'Delay (s)', 1., 'delay')]:
        name = f'{number:02d}_{kind}_bar_1x2'
        fig, axes = prior.chart.style.canvas(label)
        values = [r[metric] * scale for r in rows if r['bandwidth_mhz'] == 1]
        ticks = prior.chart.nice_ticks([0., max(values) * 1.03], bins=4, simple=True)
        panels = []
        for ax, ds in zip(axes, DATASETS):
            for i, (m, title) in enumerate(METHODS.items()):
                row = next(r for r in rows if r['bandwidth_mhz'] == 1 and r['dataset'] == ds and r['method'] == m)
                y = row[metric] * scale
                ax.bar(i, y, width=.68, color=PALETTE[i][0], edgecolor='#202020',
                       linewidth=.65, hatch=HATCHES[i], label=title)
                prior.SERIES.append(dict(figure=name, dataset=ds, method=m, kind='bar',
                                         metric=metric, scale=scale, x=[i], y=[y],
                                         bandwidth_mhz=1, hatch=HATCHES[i], color=PALETTE[i][0]))
            ax.set_xlim(-.55, len(METHODS)-.45)
            ax.set_xticks([])
            ax.grid(False, axis='x')
            prior.chart.regular_axis(ax, 'y', ticks, 0)
            assert not ax.texts
            pos = ax.get_position()
            fig.text((pos.x0 + pos.x1)/2, .015, f'({chr(97 + DATASETS.index(ds))}) {ds}',
                     ha='center', va='bottom', fontsize=14)
            panels.append(dict(dataset=ds, x=dict(kind='categorical', ticks=[],
                limits=list(ax.get_xlim()), labels=list(METHODS.values())),
                y=dict(ticks=ticks, limits=[ticks[0], ticks[-1]], labels=[str(int(v)) for v in ticks])))
        fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc='upper center',
                   bbox_to_anchor=(.5, 1.), fontsize=10, handlelength=1.6, columnspacing=1.)
        files = []
        for ext in ['png', 'pdf', 'svg']:
            path = DEST / (name + '.' + ext)
            fig.savefig(path, dpi=350, bbox_inches='tight')
            files.append(dict(file=path.name, sha256=sha(path)))
        prior.chart.plt.close(fig)
        prior.FIGURES.append(dict(name=name, title=label + ' at 1 MHz downlink bandwidth', axes=panels, files=files))


def docs(rows):
    lines = ['# Downlink bandwidth and baseline resource bars', '',
        'Current Proposed is the unchanged P-KS. Four methods: Proposed, ISCC-P2, AFL-OTA and F-SalUn. CapERA is left and AirScape right. [Learning curves and definitions](../proposed_pks/README.md) remain available.', '',
        '## Explicit bandwidth model', '',
        '`R_dl = B_dl * eta_dl`, with `eta_dl = 1 bit/s/Hz`. Downlink bandwidth is 1, 2, 3, 4, 5 MHz; the derived rates are 1, 2, 3, 4, 5 Mbit/s. Only the downlink bandwidth changes; uplink bandwidth stays 1 MHz. Transmit power stays 0.32 W and every actual full-model broadcast is charged.', '',
        'This is a declared fixed effective spectral-efficiency service model, conditional on every receiver supporting that efficiency across the sweep. It is not a measured bandwidth-throughput relationship, a Shannon-capacity calculation, or a validation of link feasibility at fixed power. There is no adaptive coding or bandwidth-dependent packet-error model. The new explicit bandwidth model overrides the inherited legacy 1 Mbps broadcast description in the physical-parameter history. An MHz axis would not have been valid without this extra assumption.', '',
        'All eight 1 MHz controls and all sixteen 3/5 MHz conditions execute the full workload: 24 new runs. The sixteen existing 2/4 Mbit/s runs are reused as equivalent 2/4 MHz conditions only under the declared eta=1 model. Original files remain unchanged; the release audit checks shared input hashes, physical settings, every broadcast duration, and recomputed costs. All eight fresh 1 MHz controls reproduce the original models, final evaluations, radio decisions and clocks. Reuse is explicit in metrics, not claimed as new execution.', '',
        '## Figures', '',
        '- 01–03: energy, active FL/FU delay and separate mission delay versus downlink bandwidth.',
        '- 04–05: energy and active-delay bar charts at the baseline 1 MHz. Hatched bars, consistent method colors, zero baseline, common legend, no bar-top numbers or extra grey text.',
        '- 06: the existing satellite/IoT/UAV map at 90% of its previous physical width and height; unchanged geometry and snapshot heads. Its provenance identifies the original snapshot run, not a newly measured bandwidth-dependent trajectory.', '',
        'Map size: 10.4 × 5.35 inches → 9.36 × 4.815 inches. Map imagery: Esri, Vantor, Earthstar Geographics, and the GIS User Community. Road data: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). The complete original attribution, source hashes and snapshot provenance are in `data/mission_map.json`.', '',
        'Energy = active communication + executed adapter computation + active hovering. Travel/IoT collection/preflight energy is excluded. Main delay is active FL/FU service time; mission delay additionally includes travel, collection and return. Same 200 total FL rounds and 225 FU steps for 45 requests. Single development seed, frozen feature caches and scoped literature adaptations; these are not full-VLM hardware measurements, statistical superiority results or certified equal-quality unlearning. Unfavorable quality and QoS-cap exceedances remain in the metrics.', '',
        '| Dataset | Method | Energy (kJ), 1 MHz | Active delay (s), 1 MHz |',
        '|---|---|---:|---:|']
    for r in rows:
        if r['bandwidth_mhz'] == 1:
            lines.append(f"| {r['dataset']} | {r['label']} | {r['total_energy_j']/1000:.3f} | {r['delay_s']:.3f} |")
    lines += ['', '## Reproduction', '', 'With the existing local caches and original rate-study arms:', '',
              '```sh', '.venv/bin/python tools/uav_convergence_v2/downlink_bandwidth_v1.py run',
              '.venv/bin/python tools/uav_convergence_v2/downlink_bandwidth_report.py',
              '.venv/bin/python tools/uav_convergence_v2/downlink_bandwidth_check.py', '```', '',
              'The frozen runner refuses to overwrite attempts. Only code, figures, aggregate metrics and hashes are published; no raw data, checkpoints or execution logs.', '']
    (DEST / 'README.md').write_text('\n'.join(lines))


def link_galleries():
    for parent in [DEST.parent, DEST.parent / 'proposed_pks']:
        relative = 'downlink_bandwidth/' if parent == DEST.parent else '../downlink_bandwidth/'
        path = parent / 'index.html'
        marker = '<!-- downlink-bandwidth-current -->'
        if marker not in path.read_text():
            text = path.read_text()
            end = text.index('</h1>') + len('</h1>')
            card = (marker + '<section><h2>Downlink bandwidth · energy/delay bars · compact map</h2>'
                    '<p><a href="' + relative + 'index.html">Open all new figures and experiment details</a></p>'
                    '<a href="' + relative + '04_energy_bar_1x2.pdf"><img src="' + relative + '04_energy_bar_1x2.png"></a></section>')
            path.write_text(text[:end] + card + text[end:])
        path = parent / 'README.md'
        marker = '<!-- downlink-bandwidth-current -->'
        if marker not in path.read_text():
            text = path.read_text()
            end = text.index('\n') + 1
            path.write_text(text[:end] + '\n' + marker + '\n[New downlink bandwidth figures, hatched energy/delay bars and 90%-size map](' + relative + 'index.html). Explicit fixed spectral efficiency; 1–5 MHz.\n' + text[end:])


def main():
    protocol = study.register()
    assert read(study.OUT / 'campaign_finished.json')['passed']
    setup()
    rows, audit = collect()
    sweep(rows)
    bars(rows)
    import compact_mission_map
    map_figure, map_audit = compact_mission_map.render(DEST)
    prior.FIGURES.append(map_figure)
    audit['map'] = map_audit
    docs(rows)
    write(DEST / 'data/audit.json', audit)
    write(DEST / 'data/protocol.json', protocol)
    write(DEST / 'data/bandwidth_model.json', study.MODEL)
    for file in [Path(__file__), Path(study.__file__), Path(compact_mission_map.__file__)]:
        prior.SOURCES[str(file.relative_to(ROOT))] = sha(file)
    export.finish(DEST, rows, 'Downlink bandwidth 1–5 MHz; hatched baseline energy/delay bars and 90%-size map.',
                  dict(compared_arms=40, new_arms=24, reused_arms=16, baseline_bandwidth_mhz=1))
    link_galleries()
    print(dict(figures=len(prior.FIGURES), arms=len(rows), destination=str(DEST)), flush=True)


if __name__ == '__main__':
    main()
