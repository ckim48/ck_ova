"""Re-export active scientific plots with common physical typography.

Replay only published plot arrays and box/bar geometry. No experiments run and
no measurements or method identities change. A fixed PDF width ensures equal
font sizes remain equal after LaTeX includes every file at width=linewidth.
"""
import copy
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import proposed_pks_report as report

ROOT = report.ROOT
BASE = ROOT/'final_final_charts'
VERSION = 'paper_typography_v1'
SIZE = (7.4, 3.735)
TICK = LEGEND = 12.
LABEL = 14.
GAP_PT = 8.
NS = {'s': 'http://www.w3.org/2000/svg'}
CATEGORY_RECTS = [(.095, .22, .385, .65), (.595, .22, .385, .65)]
CATEGORY_CAPTION_Y = .035


def category_layout_audit(fig):
    """Check physical geometry and readable single-line method labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panels = []
    for ax in fig.axes:
        labels = ax.get_xticklabels()
        assert [t.get_text() for t in labels] == list(report.METHODS.values())
        assert all(t.get_fontsize() == TICK and '\n' not in t.get_text() for t in labels)
        bounds = [t.get_window_extent(renderer) for t in labels]
        gaps = [(b.x0-a.x1)*72/fig.dpi for a, b in zip(bounds, bounds[1:])]
        assert min(gaps) >= 6., gaps
        box = ax.get_window_extent(renderer)
        panels.append(dict(width_pt=box.width*72/fig.dpi, height_pt=box.height*72/fig.dpi,
                           category_gap_pt=gaps, category_pt=TICK, one_line=True))
    assert np.allclose([p['width_pt'] for p in panels], panels[0]['width_pt'])
    assert np.allclose([p['height_pt'] for p in panels], panels[0]['height_pt'])
    return panels


def source_labels(path):
    svg = ET.parse(path).getroot()
    panels = []
    for group in svg.findall('.//s:g', NS):
        if not re.fullmatch(r'axes_\d+', group.get('id', '')):
            continue
        labels = []
        for axis in group.findall('s:g', NS):
            if not axis.get('id', '').startswith('matplotlib.axis_'):
                continue
            strings = [''.join(t.itertext()) for g in axis.findall('s:g', NS)
                       if g.get('id', '').startswith('text_') for t in g.findall('s:text', NS)]
            labels.append(strings[0] if strings else '')
        if len(labels) == 2:
            panels.append(labels)
    assert len(panels) == 2
    return panels


def render(folder, figure, series):
    if figure.get('kind') == 'rank_boxplot':
        import paper_boxplot_refine_v2
        return paper_boxplot_refine_v2.render(folder, figure)
    figure = copy.deepcopy(figure)
    chart = report.prior.chart
    name = figure['name']
    labels = source_labels(folder/(name+'.svg'))
    matched_categories = name == '11_fu_steps_energy_1x2'
    if matched_categories:
        fig = chart.plt.figure(figsize=SIZE)
        axes = [fig.add_axes(rect) for rect in CATEGORY_RECTS]
    else:
        fig, axes = chart.plt.subplots(1, 2, figsize=SIZE)
        fig.subplots_adjust(left=.115, right=.965, bottom=.22, top=.83, wspace=.46)
    for ax in axes:
        if not matched_categories:
            ax.set_box_aspect(.9)
        ax.grid(True, ls=':', lw=.5)
        ax.set_axisbelow(True)
        ax.tick_params(direction='out', pad=3, labelsize=TICK)
    records = [s for s in series if s['figure'] == name]
    legend_handles, legend_labels, legend_title = None, None, None
    kind = figure.get('kind', 'line')
    if kind == 'paired_outcome_bars':
        legend_handles = [Patch(facecolor=c, edgecolor='#202020', linewidth=.65, hatch=h)
                          for c, h in zip(['#4C72B0', '#E6B45A', '#C65365'], ['//', '', '..'])]
        legend_labels = ['Improved', 'Tied', 'Worsened']
        legend_title = 'Proposed vs. each comparator at 650 kJ'
    for ax, panel, label_pair in zip(axes, figure['axes'], labels):
        ds = panel['dataset']
        selected = [s for s in records if s['dataset'] == ds]
        if kind == 'paired_outcome_bars':
            for b in [b for b in figure['bars'] if b['dataset'] == ds]:
                artist = ax.bar(b['x'], b['height'], bottom=b['bottom'], width=.62,
                    color=b['color'], hatch=b['hatch'], edgecolor='#202020', linewidth=.65)[0]
                assert artist.get_height() == b['height'] and artist.get_y() == b['bottom']
        elif kind == 'rank_boxplot':
            for i, m in enumerate(report.METHODS):
                s = next(b for b in figure['boxes'] if (b['dataset'], b['method']) == (ds, m))
                stats = dict(q1=s['q1'], med=s['median'], q3=s['q3'], whislo=s['whisker_low'],
                             whishi=s['whisker_high'], fliers=s['fliers'])
                made = ax.bxp([stats], positions=[i], widths=.52, patch_artist=True,
                    showfliers=True, showmeans=False, manage_ticks=False,
                    boxprops=dict(facecolor=s['color'], edgecolor='#202020', linewidth=.8, alpha=.8),
                    medianprops=dict(color='#202020', linewidth=1.2),
                    whiskerprops=dict(color='#202020', linewidth=.75), capprops=dict(color='#202020', linewidth=.75),
                    flierprops=dict(marker='o', markersize=1.8, markerfacecolor=s['color'],
                                    markeredgecolor=s['color'], markeredgewidth=.2, alpha=.55))
                assert made['fliers'][0].get_ydata().tolist() == s['fliers']
        elif selected and selected[0].get('kind') in ['bar', 'paired_bar']:
            for s in selected:
                paired = s['kind'] == 'paired_bar'
                positions = s['bar_positions'] if paired else s['x']
                hatches = s['hatches'] if paired else [s['hatch']]
                for x, y, hatch in zip(positions, s['y'], hatches):
                    b = ax.bar(x, y, width=.33, color=s['color'], edgecolor='#202020', linewidth=.65, hatch=hatch)[0]
                    assert b.get_height() == y and np.isclose(b.get_x()+b.get_width()/2, x)
            legend_handles = [Patch(facecolor='white', edgecolor='#202020', linewidth=.65, hatch=h)
                              for h in (['//', '..'] if paired else ['//', '..'])]
            legend_labels = ['1 step / request', '9 steps / request'] if paired else ['1 MHz', '5 MHz']
        else:
            assert len(selected) == 4, (name, ds)
            for i, (method, method_label) in enumerate(report.METHODS.items()):
                s = next(s for s in selected if s['method'] == method)
                x, y = s.get('display_x', s['x']), s.get('display_y', s['y'])
                line = chart.style.curve(ax, x, y, i, method_label)
                if 'drawstyle' in s:
                    line.set_drawstyle(s['drawstyle'])
                if 'marker_indices' in s:
                    line.set_markevery(s['marker_indices'])
                if 'marker_x' in s:
                    line.set_markevery([])
                    ax.plot(s['marker_x'], s['marker_y'], linestyle='None', marker=report.PALETTE[i][1],
                            color=report.PALETTE[i][0], markerfacecolor='white', markersize=line.get_markersize(),
                            markeredgewidth=line.get_markeredgewidth(), label='_nolegend_')
                assert line.get_xdata().tolist() == x and line.get_ydata().tolist() == y
        for dim in ['x', 'y']:
            a = panel[dim]
            if dim == 'x' and a.get('kind') == 'categorical' and len(a['ticks']) == 4:
                # Recover space at the edges instead of shrinking method names.
                # The outer paired bars end at -0.35 and3.35, inside this range.
                a['limits'] = [-.4, 3.4]
            getattr(ax, 'set_'+dim+'lim')(a['limits'])
            getattr(ax, 'set_'+dim+'ticks')(a['ticks'])
            if a.get('kind') == 'categorical':
                category_size = 12. if name == '11_fu_steps_energy_1x2' else LABEL
                displayed = [s.replace('-', '-\n', 1) if len(a['ticks']) == 4 and category_size == LABEL else s for s in a['labels']]
                getattr(ax, 'set_'+dim+'ticklabels')(displayed, fontsize=category_size)
                ax.tick_params(axis=dim, labelsize=category_size, length=0, pad=6)
                ax.grid(False, axis=dim)
            else:
                precision = max(len(s.split('.')[-1]) if '.' in s else 0 for s in a['labels'])
                getattr(ax, dim+'axis').set_major_formatter(FormatStrFormatter(f'%.{precision}f'))
            assert getattr(ax, 'get_'+dim+'lim')() == tuple(a['limits'])
        ax.set_xlabel(label_pair[0], fontsize=LABEL)
        ax.set_ylabel(label_pair[1], fontsize=LABEL, labelpad=4)
        pos = ax.get_position()
        fig.text((pos.x0+pos.x1)/2, CATEGORY_CAPTION_Y if matched_categories else .015, f'({chr(97+report.DATASETS.index(ds))}) {ds}',
                 ha='center', va='bottom', fontsize=LABEL)
    if kind != 'rank_boxplot':
        if legend_handles is None:
            legend_handles, legend_labels = axes[0].get_legend_handles_labels()
        top = max(ax.get_position().y1 for ax in axes)
        legend = fig.legend(legend_handles, legend_labels, ncol=len(legend_labels),
            loc='lower center', bbox_to_anchor=(.5, top+GAP_PT/(72*SIZE[1])), borderaxespad=0,
            fontsize=LEGEND, title=legend_title, title_fontsize=11,
            handlelength=1.6, handletextpad=.45, columnspacing=1., frameon=False)
    else:
        legend = None
        top = max(ax.get_position().y1 for ax in axes)
        fig.text(.5, top+GAP_PT/(72*SIZE[1]), 'Ground-truth video rank at 650 kJ (lower is better)',
                 ha='center', va='bottom', fontsize=LEGEND)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    font_audit, gap = [], None
    for ax, panel in zip(axes, figure['axes']):
        x_size = (12. if name == '11_fu_steps_energy_1x2' else LABEL) if panel['x'].get('kind') == 'categorical' else TICK
        assert all(t.get_fontsize() == x_size for t in ax.get_xticklabels())
        assert all(t.get_fontsize() == TICK for t in ax.get_yticklabels())
        assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == LABEL
        bounds = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
        assert all(a.x1+1 < b.x0 for a, b in zip(bounds, bounds[1:])), (name, 'x labels overlap')
        font_audit.append(dict(x_tick_pt=x_size, y_tick_pt=TICK, x_label_pt=LABEL, y_label_pt=LABEL,
                               category_names_match_axis_title=panel['x'].get('kind') == 'categorical' and x_size == LABEL))
    if legend is not None:
        assert all(t.get_fontsize() == LEGEND for t in legend.get_texts())
        box = legend.get_window_extent(renderer)
        gap = (box.y0-max(ax.get_window_extent(renderer).y1 for ax in axes))*72/fig.dpi
        assert abs(gap-GAP_PT) < .1, (name, gap)
        assert box.x0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, (name, box)
    # Native text extents must fit the fixed canvas; no automatic tight crop.
    visible_text = list(fig.texts)
    for ax in axes:
        visible_text += [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label]
    if legend is not None:
        visible_text += list(legend.get_texts())+[legend.get_title()]
    for t in visible_text:
        if not t.get_text():
            continue
        box = t.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, (name, t.get_text(), box)
    layout = category_layout_audit(fig) if matched_categories else None
    files = []
    for ext in ['pdf', 'png', 'svg']:
        path = folder/(name+'.'+ext)
        fig.savefig(path, dpi=350, bbox_inches=None)
        files.append(dict(file=path.name, sha256=report.sha(path)))
    chart.plt.close(fig)
    result = copy.deepcopy(figure)
    result['files'] = files
    result['typography'] = dict(version=VERSION, figure_inches=list(SIZE), axes=font_audit,
                               legend_pt=LEGEND if legend is not None else None,
                               legend_gap_pt=gap, fixed_pdf_width_pt=SIZE[0]*72.)
    if matched_categories:
        result['typography'].update(category_layout_revision='matched_rank_fu_v1', physical_panels=layout)
    return result


def main():
    report.setup()
    chart = report.prior.chart
    chart.plt.rcParams['savefig.bbox'] = None
    chart.plt.rcParams['hatch.linewidth'] = .55
    locations = [(BASE/'proposed_pks', 'manifest.json', 'plot_series.json'),
                 (BASE/'downlink_bandwidth', 'manifest.json', 'plot_series.json'),
                 (BASE/'downlink_bandwidth', 'tgcn_manifest.json', 'tgcn_plot_series.json')]
    documents = [(folder, name, report.read(folder/'data'/name), report.read(folder/'data'/series))
                 for folder, name, series in locations]
    measurements = {str(p.relative_to(ROOT)): report.sha(p) for folder, _, _ in locations
                    for p in (folder/'data').iterdir() if p.suffix in ['.json', '.csv'] and
                    any(s in p.name for s in ['metrics', 'series', 'query_ranks', 'paired_counts',
                                             'summary', 'box_stats', 'comparator_outcomes'])}
    before, changed = {}, []
    for folder, manifest_name, manifest, series in documents:
        for index, figure in enumerate(manifest['figures']):
            if len(figure.get('axes', [])) != 2:
                continue  # Maps, tables and historical example sheets have no plot legends.
            for item in figure['files']:
                before[str((folder/item['file']).relative_to(ROOT))] = item['sha256']
            result = render(folder, figure, series)
            if result.get('kind') == 'rank_boxplot':
                style_source = Path(__file__).with_name('paper_boxplot_refine_v2.py')
                manifest['source_sha256'][str(style_source.relative_to(ROOT))] = report.sha(style_source)
            manifest['figures'][index] = result
            changed.append(dict(folder=str(folder.relative_to(ROOT)), name=result['name'], typography=result['typography']))
            print('Restyled '+str(folder.relative_to(BASE))+'/'+figure['name'], flush=True)
        manifest['source_sha256'][str(Path(__file__).relative_to(ROOT))] = report.sha(Path(__file__))
        manifest['typography_revision'] = VERSION
    down = BASE/'downlink_bandwidth'
    base_manifest = documents[1][2]
    tgcn_manifest = documents[2][2]
    tgcn_manifest.setdefault('original_figures_before_typography', tgcn_manifest['original_figures'].copy())
    tgcn_manifest['original_figures'] = {f: report.sha(down/f) for f in tgcn_manifest['original_figures']}
    for folder, name, manifest, _ in documents:
        report.write(folder/'data'/name, manifest)
    assert all(report.sha(ROOT/p) == digest for p, digest in measurements.items())
    target = BASE/'figures'
    target.mkdir(exist_ok=True)
    aliases = [(down/'11_fu_steps_energy_1x2', 'fu_steps_energy_1x2'),
               (BASE/'proposed_pks/01_round_loss_1x2', '01_round_loss_1x2'),
               (BASE/'proposed_pks/05_time_mr_1x2', '05_time_mr_1x2')]
    for source, stem in aliases:
        for ext in ['pdf', 'png', 'svg']:
            shutil.copyfile(source.with_suffix('.'+ext), target/(stem+'.'+ext))
    report.write(BASE/'data_typography_v1.json', dict(passed=True, version=VERSION,
        figures=len(changed), fixed_pdf_width_pt=SIZE[0]*72, tick_font_pt=TICK, axis_label_font_pt=LABEL,
        category_font_pt=LABEL, category_font_overrides={'11_fu_steps_energy_1x2': 12.},
        legend_font_pt=LEGEND, legend_gap_pt=GAP_PT, before_artifact_sha256=before,
        unchanged_measurement_sha256=measurements, updated=changed,
        source_sha256={str(Path(__file__).relative_to(ROOT)): report.sha(Path(__file__))}))
    (target/'README.md').write_text('# Matching paper figures\n\n'
        'These aliases use the current P-KS experiments and match the requested LaTeX filenames. '
        'All three PDFs have the same 532.8 pt width, 12 pt numeric ticks and legends, and 14 pt axis titles. '
        'FU method names use 12 pt on one line; other four-method category names wrap at the hyphen at 14 pt. '
        'Legends sit 8 pt above the plots. Thus width=\\linewidth gives matching apparent font sizes.\n\n'
        '- fu_steps_energy_1x2: current 11, paired 1-step and 9-step FU workloads.\n'
        '- 01_round_loss_1x2: current P-KS round validation loss.\n'
        '- 05_time_mr_1x2: current P-KS retained mR under the shared time budget.\n\n'
        'The active proposed_pks and downlink_bandwidth chart files have also been updated. '
        'Maps, tables and earlier qualitative sheets keep their existing layout. '
        'Reapply this presentation pass after rebuilding plots: '
        '`.venv/bin/python tools/uav_convergence_v2/paper_typography_v1.py`.\n')
    print(dict(passed=True, styled_figures=len(changed), common_width_pt=SIZE[0]*72), flush=True)


if __name__ == '__main__':
    main()
