"""Plot the existing energy-to-mR targets for Proposed versus AFL-OTA."""
import csv
import math
from pathlib import Path

from matplotlib.ticker import FormatStrFormatter
import proposed_pks_report as report
import paper_typography_v1 as style

ROOT = report.ROOT
DEST = ROOT/'final_final_charts/downlink_bandwidth'
NAME = '20_energy_to_target_mr_1x2'
METHODS = ['gate_select16', 'afl']


def main():
    data = DEST/'data'
    inputs = [data/'energy_utility_checkpoints.json', data/'energy_to_mr_targets.json']
    source_hashes = {str(p.relative_to(ROOT)): report.sha(p) for p in inputs}
    old_figures = {p.name: report.sha(p) for p in DEST.iterdir()
                   if p.suffix in ['.pdf', '.png', '.svg'] and p.stem != NAME}
    traces, targets = map(report.read, inputs)
    # Verify every existing target, including unreached comparator targets.
    for row in targets:
        points = sorted((p for p in traces if (p['dataset'], p['method']) ==
                         (row['dataset'], row['method'])), key=lambda p: p['energy_kJ'])
        assert points and all(p['energy_kJ'] <= 650 for p in points)
        first = next((p for p in points if p['FU_completed'] == 45 and
                      p['retained_mR_percent'] >= row['target_mR_percent']), None)
        assert row['reached'] == (first is not None)
        for metric in ['energy_kJ', 'active_s']:
            assert row[metric] == (None if first is None else first[metric])
    comparisons, series = [], []
    for ds in report.DATASETS:
        values = sorted({r['target_mR_percent'] for r in targets if r['dataset'] == ds})
        for target in values:
            pair = {m: next(r for r in targets if (r['dataset'], r['method'], r['target_mR_percent']) ==
                           (ds, m, target)) for m in METHODS}
            assert all(r['reached'] for r in pair.values())
            p, a = [pair[m] for m in METHODS]
            comparisons.append(dict(dataset=ds, target_mR_percent=target,
                proposed_energy_kJ=p['energy_kJ'], afl_energy_kJ=a['energy_kJ'],
                energy_reduction_percent=100*(1-p['energy_kJ']/a['energy_kJ']),
                proposed_active_s=p['active_s'], afl_active_s=a['active_s'],
                active_time_reduction_percent=100*(1-p['active_s']/a['active_s'])))
        for method in METHODS:
            selected = sorted((r for r in targets if (r['dataset'], r['method']) == (ds, method)),
                              key=lambda r: r['target_mR_percent'])
            series.append(dict(dataset=ds, method=method,
                x=[r['target_mR_percent'] for r in selected], y=[r['energy_kJ'] for r in selected]))
    report.setup()
    plt = report.prior.chart.plt
    plt.rcParams['savefig.bbox'] = None
    fig, axes = plt.subplots(1, 2, figsize=style.SIZE)
    fig.subplots_adjust(left=.115, right=.965, bottom=.22, top=.83, wspace=.46)
    panels = []
    for ax, ds in zip(axes, report.DATASETS):
        ax.set_box_aspect(.9)
        ax.grid(True, ls=':', lw=.5)
        ax.set_axisbelow(True)
        ax.tick_params(direction='out', pad=3, labelsize=style.TICK)
        for method in METHODS:
            row = next(r for r in series if (r['dataset'], r['method']) == (ds, method))
            index = list(report.METHODS).index(method)
            line = report.prior.chart.style.curve(ax, row['x'], row['y'], index, report.METHODS[method])
            line.set_markevery(1)
            line.set_clip_on(False)
            assert line.get_xdata().tolist() == row['x'] and line.get_ydata().tolist() == row['y']
        ticks = row['x']
        ax.set_xlim(ticks[0], ticks[-1]); ax.set_xticks(ticks)
        ax.set_ylim(0, 600); ax.set_yticks(range(0, 601, 100))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_xlabel('Target retained mR (%)', fontsize=style.LABEL)
        ax.set_ylabel('Energy to target (kJ)', fontsize=style.LABEL, labelpad=4)
        pos = ax.get_position()
        fig.text((pos.x0+pos.x1)/2, .015, f'({chr(97+report.DATASETS.index(ds))}) {ds}',
                 ha='center', va='bottom', fontsize=style.LABEL)
        panels.append(dict(dataset=ds, x_ticks=ticks, y_ticks=list(range(0, 601, 100))))
    handles, labels = axes[0].get_legend_handles_labels()
    top = max(ax.get_position().y1 for ax in axes)
    legend = fig.legend(handles, labels, ncol=2, loc='lower center',
        bbox_to_anchor=(.5, top+style.GAP_PT/(72*style.SIZE[1])), borderaxespad=0,
        fontsize=style.LEGEND, handlelength=1.6, handletextpad=.45, columnspacing=1., frameon=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap = (legend.get_window_extent(renderer).y0-max(ax.get_window_extent(renderer).y1 for ax in axes))*72/fig.dpi
    assert math.isclose(gap, style.GAP_PT, abs_tol=.1)
    for t in [*fig.texts, *legend.get_texts(), *[t for ax in axes for t in
              [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]]]:
        box = t.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1
    files = []
    for ext in ['pdf', 'png', 'svg']:
        path = DEST/(NAME+'.'+ext)
        fig.savefig(path, dpi=350, bbox_inches=None)
        files.append(dict(file=path.name, sha256=report.sha(path)))
    plt.close(fig)
    csv_path = data/'energy_target_afl_comparison.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader(); writer.writerows(comparisons)
    caption = ('Energy required by Proposed and AFL-OTA to first reach the previously reported retained-mR targets, '
        'after all 45 prescribed FU procedures have completed. Each marker is an actual evaluated checkpoint; '
        'connecting lines are visual guides. The plotted targets are 18%, 20%, and 22% for CapERA, and 14%, 15%, '
        'and 16% for AirScape. The maximum observation budget is 650 kJ. ISCC-P2 and F-SalUn do not reach these '
        'targets within this budget and retain their unreached entries in energy_to_mr_targets.csv. '
        'This is a focused comparison of the two OTA methods. Energy includes active communication, adapter '
        'computation, and hovering. Threshold crossing is first observed attainment, not sustained convergence. '
        'FU procedure completion does not imply equal forgetting quality: KL is not matched or constrained here. '
        'These are descriptive results from one development seed, not a statistical superiority claim.\n')
    (DEST/(NAME+'_caption.txt')).write_text(caption)
    notes = DEST/'TGCN_ADDITIONS.md'
    marker = '<!-- energy-target-afl -->'
    end = '<!-- /energy-target-afl -->'
    text = notes.read_text()
    if marker in text:
        before, tail = text.split(marker, 1)
        text = before+tail.split(end, 1)[1].lstrip('\n')
    section = marker+'\n\n## 20: Energy to a shared retained-mR target\n\n'+caption+'\n'
    section += f'[Figure PDF]({NAME}.pdf) · [Exact energy and time comparisons](data/energy_target_afl_comparison.csv)\n\n'
    section += '| Dataset | Target mR (%) | Proposed (kJ) | AFL-OTA (kJ) | Energy reduction |\n|---|---:|---:|---:|---:|\n'
    for r in comparisons:
        section += f"| {r['dataset']} | {r['target_mR_percent']:.0f} | {r['proposed_energy_kJ']:.2f} | {r['afl_energy_kJ']:.2f} | {r['energy_reduction_percent']:.2f}% |\n"
    section += '\nReproduce with `.venv/bin/python tools/uav_convergence_v2/energy_target_comparison_v1.py`.\n\n'+end+'\n\n'
    notes.write_text(text.replace('## Reproduction\n', section+'## Reproduction\n', 1))
    index = DEST/'index.html'
    text = index.read_text()
    if marker in text:
        before, tail = text.split(marker, 1)
        text = before+tail.split(end, 1)[1]
    section = marker+f'<section><h2>Proposed versus AFL-OTA: energy to a shared retained-mR target</h2><a href="{NAME}.pdf"><img loading="lazy" src="{NAME}.png"></a><p><a href="{NAME}_caption.txt">Conditions and interpretation</a></p></section>'+end
    index.write_text(text.replace('<!-- /tgcn-additions -->', '<!-- /tgcn-additions -->'+section, 1))
    assert all(report.sha(ROOT/p) == digest for p, digest in source_hashes.items())
    assert all(report.sha(DEST/p) == digest for p, digest in old_figures.items())
    report.write(data/'energy_target_afl_comparison_manifest.json', dict(passed=True, figure=NAME,
        files=files, series=series, panels=panels, comparisons=comparisons, caption=caption,
        checked_targets=len(targets), all_existing_targets_checked=True, all_previous_figures_unchanged=True,
        matched_KL=False, new_training_runs=0, figure_inches=list(style.SIZE),
        axis_label_pt=style.LABEL, numeric_tick_pt=style.TICK, legend_pt=style.LEGEND, legend_gap_pt=gap,
        input_sha256=source_hashes, source_sha256={str(p.relative_to(ROOT)):report.sha(p)
            for p in [Path(__file__), Path(report.__file__), Path(style.__file__)]}))
    print(dict(passed=True, figure=NAME, comparisons=comparisons, new_training_runs=0), flush=True)


if __name__ == '__main__':
    main()
