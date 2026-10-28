"""Title-free rank boxes with a separate detail strip; no data selection."""
import copy
from pathlib import Path

import numpy as np
import matplotlib.patheffects as patheffects
from matplotlib.colors import to_rgba
from matplotlib.ticker import FormatStrFormatter
import proposed_pks_report as report

ROOT = report.ROOT
BASE = ROOT/'final_final_charts'
DEST = BASE/'downlink_bandwidth'
SIZE = (7.4, 3.735)
NAME = '19_all_method_rank_boxplot_1x2'


def render(folder, figure):
    plt = report.prior.chart.plt
    fig = plt.figure(figsize=SIZE, facecolor='white')
    result = copy.deepcopy(figure)
    main_axes, details, draw_checks = [], [], []
    for index, ds in enumerate(report.DATASETS):
        left = [.115, .625][index]
        main = fig.add_axes([left, .22, .34, .44])
        detail = fig.add_axes([left, .77, .34, .205])
        main_axes.append(main)
        panel = next(p for p in result['axes'] if p['dataset'] == ds)
        main.set_ylim(panel['y']['limits']); main.set_yticks(panel['y']['ticks'])
        detail.set_ylim(0, 100); detail.set_yticks([0, 50, 100])
        for ax in [main, detail]:
            ax.set_xlim(-.4, 3.4)
            ax.set_axisbelow(True)
            ax.grid(True, axis='y', color='#D6DCE3', linestyle=':', linewidth=.45)
            ax.tick_params(axis='y', labelsize=12, direction='out', pad=3, length=2.5)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        main.set_ylabel(r'GT rank ($\downarrow$)', fontsize=14, labelpad=4)
        main.set_xticks(range(4), [s.replace('-', '-\n', 1) for s in report.METHODS.values()], fontsize=14)
        main.tick_params(axis='x', length=0, pad=6)
        detail.set_xticks([])
        # The shared tint identifies the same 0--100 interval in both views.
        main.axhspan(0, 100, facecolor='#F2F5F9', edgecolor='none', zorder=-2)
        detail.set_facecolor('#F2F5F9')
        for side in ['top', 'right']:
            main.spines[side].set_visible(False)
        for spine in detail.spines.values():
            spine.set_color('#9CA9B9'); spine.set_linewidth(.6)
        for method_index, method in enumerate(report.METHODS):
            s = next(b for b in result['boxes'] if (b['dataset'], b['method']) == (ds, method))
            color, marker, _ = report.PALETTE[method_index]
            assert color == s['color']
            stats = dict(q1=s['q1'], med=s['median'], q3=s['q3'],
                         whislo=s['whisker_low'], whishi=s['whisker_high'], fliers=s['fliers'])
            y = np.asarray(s['fliers'], dtype=float)
            # Deterministic horizontal spreading only. Every rank stays exact.
            offset = (((np.arange(len(y))+1)*.6180339887498949) % 1-.5)*.13
            for ax in [main, detail]:
                made = ax.bxp([stats], positions=[method_index], widths=.50, manage_ticks=False,
                    showfliers=False, showmeans=False, patch_artist=True,
                    boxprops=dict(facecolor=color, edgecolor='#202020', linewidth=.8),
                    medianprops=dict(color='#202020', linewidth=1.2, zorder=4),
                    whiskerprops=dict(color=color, linewidth=1.05),
                    capprops=dict(color=color, linewidth=1.05))
                # Equal styling across methods: white halo keeps each median
                # visible on the same opaque colors used by the paper bars.
                made['medians'][0].set_path_effects([
                    patheffects.Stroke(linewidth=2.8, foreground='white'),
                    patheffects.Normal()])
                median_point, = ax.plot(method_index, s['median'], linestyle='None',
                    marker=marker, markersize=4.2, markerfacecolor='white',
                    markeredgecolor=color, markeredgewidth=1.05, zorder=5)
                assert median_point.get_ydata().tolist() == [s['median']]
                assert made['boxes'][0].get_facecolor() == to_rgba(color)
                # Replace coincident default fliers with all original observations.
                dots = ax.scatter(method_index+offset, y, s=4.0, facecolors='none',
                                  edgecolors=to_rgba(s['color'], .55), linewidths=.38, zorder=3)
                assert np.array_equal(np.asarray(dots.get_offsets())[:, 1], y)
                assert made['medians'][0].get_ydata().tolist() == [s['median'], s['median']]
            draw_checks.append(dict(dataset=ds, method=method, original_fliers=len(y),
                displayed_full_range_fliers=len(y), displayed_detail_fliers=int(sum(y <= 100)),
                vertical_values_unchanged=True, horizontal_spread_half_width=.065,
                box_facecolor=color, box_face_alpha=1., median_marker=marker,
                median_marker_y=s['median'], median_marker_size_pt=4.2))
        # bxp may adjust data bounds even with manage_ticks=False; restore them.
        main.set_xlim(-.4, 3.4); main.set_ylim(panel['y']['limits'])
        detail.set_xlim(-.4, 3.4); detail.set_ylim(0, 100)
        details.append(dict(dataset=ds, x_limits=[-.4, 3.4], y_limits=[0, 100],
            y_ticks=[0, 50, 100], same_statistics_as_main=True, separate_nonoverlapping_axes=True))
        fig.text(left+.17, .015, f'({chr(97+index)}) {ds}', ha='center', va='bottom', fontsize=14)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for t in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.yaxis.label]:
            if t.get_text():
                bbox = t.get_window_extent(renderer)
                assert bbox.x0 >= -1 and bbox.y0 >= -1 and bbox.x1 <= fig.bbox.width+1 and bbox.y1 <= fig.bbox.height+1, t.get_text()
    assert len(fig.axes) == 4 and fig._suptitle is None
    for i in range(0, 4, 2):
        assert fig.axes[i].get_position().y1 < fig.axes[i+1].get_position().y0
    files = []
    for ext in ['pdf', 'png', 'svg']:
        path = folder/(figure['name']+'.'+ext)
        fig.savefig(path, dpi=350, bbox_inches=None)
        files.append(dict(file=path.name, sha256=report.sha(path)))
    plt.close(fig)
    result.update(files=files, display_revision='title_free_detail_v2', title_removed=True,
        detail_axes=details, draw_checks=draw_checks,
        palette_revision='opaque_shared_palette_medians', lower_rank_is_better=True,
        caption='GT-rank distributions of all retained text queries at the same 650 kJ budget. Lower ranks are better. Each lower axis shows the full 0--700 range with every outlier; the separate upper strip repeats exactly the same box statistics in the 0--100 interval. Shared light shading marks that interval. Opaque box colors and median markers match the method palette in the bar and line charts. Every method uses the same median emphasis; hollow outlier markers receive deterministic horizontal spreading only. Boxes are query distributions, not confidence intervals over independent training seeds.')
    result.setdefault('typography', {}).update(version='paper_typography_v1', figure_inches=list(SIZE),
        legend_pt=None, legend_gap_pt=None, fixed_pdf_width_pt=532.8, title_removed=True,
        detail_numeric_tick_pt=12)
    return result


def main(only_rank=False):
    import paper_typography_v1 as typography
    report.setup()
    report.prior.chart.plt.rcParams['savefig.bbox'] = None
    report.prior.chart.plt.rcParams['hatch.linewidth'] = .55
    path = DEST/'data/tgcn_manifest.json'
    manifest = report.read(path)
    names = [NAME] if only_rank else ['11_fu_steps_energy_1x2', NAME]
    before = {f['file']: report.sha(DEST/f['file']) for figure in manifest['figures'] for f in figure['files']}
    measurements = {str(p.relative_to(ROOT)): report.sha(p) for p in (DEST/'data').iterdir()
                    if p.suffix in ['.csv', '.json'] and any(s in p.name for s in
                    ['metrics', 'series', 'query_ranks', 'paired_counts', 'summary', 'box_stats', 'comparator_outcomes'])}
    series = report.read(DEST/'data/tgcn_plot_series.json')
    for i, figure in enumerate(manifest['figures']):
        if figure['name'] in names:
            manifest['figures'][i] = typography.render(DEST, figure, series)
    # Record the updated presentation code without touching other rendered plots.
    for folder, file in [(BASE/'proposed_pks', 'manifest.json'), (DEST, 'manifest.json'), (DEST, 'tgcn_manifest.json')]:
        document = manifest if file == 'tgcn_manifest.json' else report.read(folder/'data'/file)
        document['source_sha256'][str(Path(typography.__file__).relative_to(ROOT))] = report.sha(Path(typography.__file__))
        if file == 'tgcn_manifest.json':
            document['source_sha256'][str(Path(__file__).relative_to(ROOT))] = report.sha(Path(__file__))
        report.write(folder/'data'/file, document)
    import shutil
    if not only_rank:
        for ext in ['pdf', 'png', 'svg']:
            shutil.copyfile(DEST/('11_fu_steps_energy_1x2.'+ext), BASE/'figures'/('fu_steps_energy_1x2.'+ext))
    assert all(report.sha(ROOT/p) == digest for p, digest in measurements.items())
    assert all(report.sha(DEST/f) == digest for f, digest in before.items() if not any(f.startswith(name+'.') for name in names))
    audit = report.read(BASE/'data_typography_v1.json')
    for row in audit['updated']:
        if row['name'] in names and row['folder'].endswith('downlink_bandwidth'):
            row['typography'] = next(f for f in manifest['figures'] if f['name'] == row['name'])['typography']
    audit['category_font_overrides'] = {'11_fu_steps_energy_1x2': 12.}
    audit['source_sha256'][str(Path(typography.__file__).relative_to(ROOT))] = report.sha(Path(typography.__file__))
    audit['source_sha256'][str(Path(__file__).relative_to(ROOT))] = report.sha(Path(__file__))
    report.write(BASE/'data_typography_v1.json', audit)
    checks = report.read(BASE/'data_typography_v1_checks.json')
    checks.pop('matched_FU_category_and_reference_axis_title_pt', None)
    checks.update(FU_category_pt=12, reference_axis_title_pt=14, FU_category_override_requested=True,
                  boxplot_title_removed=True, boxplot_full_range_and_detail_verified=True)
    report.write(BASE/'data_typography_v1_checks.json', checks)
    readme = BASE/'figures/README.md'
    text = readme.read_text().replace('14 pt category names/axis titles', '14 pt axis titles and category names, with the requested FU category exception at 12 pt')
    text = text.replace('Four-method category names wrap at the hyphen to preserve font size without overlap.',
                        'FU method names use 12 pt on one line; other four-method category names wrap at the hyphen at 14 pt.')
    text = text.replace('at12 pt', 'at 12 pt').replace('use12 pt', 'use 12 pt').replace('at14 pt', 'at 14 pt')
    readme.write_text(text)
    notes = DEST/'TGCN_ADDITIONS.md'
    marker = '<!-- rank-detail-v2 -->'
    text = notes.read_text().split(marker)[0].rstrip()
    notes.write_text(text+'\n\n'+marker+'\n\n'
        'Figure 19 has no overall title. The lower axes retain the full 0–700 rank range; '
        'the separate upper strips show the same box statistics over 0–100. Shared shading '
        'identifies this interval. Every outlier is retained as a hollow marker, with only '
        'horizontal spreading to reduce overlap. Box fills now use the exact opaque colors '
        'from the method palette in the bar and line charts. Each method has the same median '
        'line emphasis and a matching line-chart marker; lower GT ranks are better. '
        'On AirScape, the Proposed and AFL-OTA medians are 35.5 and 37, while their '
        '75th percentiles are 95.25 and 89.25; neither distribution is uniformly better. '
        'Figure 11 uses 12 pt method names on one line. '
        'Reapply these presentation changes with '
        '`.venv/bin/python tools/uav_convergence_v2/paper_boxplot_refine_v2.py`.\n')
    report.write(DEST/'data/boxplot_refinement_v2_checks.json', dict(passed=True,
        full_range=[0, 700], detail_range=[0, 100], title_removed=True,
        opaque_shared_palette=True, same_median_emphasis_for_all_methods=True,
        before_artifact_sha256=before, unchanged_measurement_sha256=measurements,
        draw_checks=next(f for f in manifest['figures'] if f['name'] == NAME)['draw_checks']))
    print(dict(passed=True, updated=names, no_new_experiments=True), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-rank', action='store_true')
    main(only_rank=parser.parse_args().only_rank)
