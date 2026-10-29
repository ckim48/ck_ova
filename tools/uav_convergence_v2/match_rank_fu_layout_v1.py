"""Match the limited-energy rank and FU bar layouts using published data only."""
import copy
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import paper_typography_v1 as typography
import paper_boxplot_refine_v2 as rank

report = typography.report
ROOT, BASE, DEST = report.ROOT, typography.BASE, rank.DEST
BAR = '11_fu_steps_energy_1x2'
RANK = rank.NAME


def main(only_rank=False):
    report.setup()
    report.prior.chart.plt.rcParams.update({'savefig.bbox': None, 'hatch.linewidth': .55})
    rank.SIZE = typography.SIZE
    limited = DEST/'limited_energy'
    # Hash the actual measurements, including the current methodology pilot.
    measurements = [DEST/'data/tgcn_plot_series.json', limited/'box_statistics.json',
                    limited/'paired_outcomes.json', limited/'progress.json', limited/'progress.csv']
    for budget in [200, 300, 400]:
        measurements += list((limited/f'{budget}kJ').glob('*query_ranks.*'))
        measurements += list((limited/f'{budget}kJ').glob('progress.*'))
    measurements += list((BASE/'manuscript_iq_pilot').glob('*'))
    measurements = [p for p in measurements if p.is_file()]
    before = {str(p.relative_to(ROOT)): report.sha(p) for p in measurements}
    original_650 = {str((DEST/(RANK+'.'+e)).relative_to(ROOT)): report.sha(DEST/(RANK+'.'+e))
                    for e in ['pdf', 'png', 'svg']}
    sources = {str(p.relative_to(ROOT)): report.sha(p) for p in
               [Path(__file__), Path(typography.__file__), Path(rank.__file__)]}
    updated = []
    for budget in [200, 300, 400]:
        folder = limited/f'{budget}kJ'
        old = report.read(folder/'manifest.json')
        result = rank.render(folder, old)
        assert old['boxes'] == result['boxes'] and old['axes'] == result['axes']
        # The shared renderer's historic default caption refers to 650 kJ.
        result['caption'] = old['caption']
        result['presentation_source_sha256'] = sources
        report.write(folder/'manifest.json', result)
        updated.append(dict(folder=str(folder.relative_to(ROOT)), name=RANK,
                            typography=result['typography'], files=result['files']))

    path = DEST/'data/tgcn_manifest.json'
    manifest = report.read(path)
    series = report.read(DEST/'data/tgcn_plot_series.json')
    other_figures = {f['file']: report.sha(DEST/f['file']) for figure in manifest['figures']
                     if only_rank or figure['name'] != BAR for f in figure['files']}
    index = next(i for i, f in enumerate(manifest['figures']) if f['name'] == BAR)
    old = copy.deepcopy(manifest['figures'][index])
    result = old if only_rank else typography.render(DEST, old, series)
    assert old['axes'] == result['axes']
    manifest['figures'][index] = result
    manifest['source_sha256'].update(sources)
    report.write(path, manifest)
    updated.append(dict(folder=str(DEST.relative_to(ROOT)), name=BAR,
                        typography=result['typography'], files=result['files']))

    # Publish unambiguous LaTeX aliases together. The rank alias is the 300 kJ case.
    aliases = [(limited/'300kJ'/RANK, 'all_method_rank_boxplot_1x2'),
               (DEST/BAR, 'fu_steps_energy_1x2')]
    target = BASE/'figures'
    for source, name in aliases:
        for ext in ['pdf', 'png', 'svg']:
            shutil.copyfile(source.with_suffix('.'+ext), target/(name+'.'+ext))
    shutil.copyfile(limited/'300kJ'/(RANK+'_caption.txt'),
                    target/'all_method_rank_boxplot_1x2_caption.txt')

    # Inspect the exported vector geometry as well as the Matplotlib layout.
    vector_panels = []
    for row in updated:
        svg = ET.parse(ROOT/row['folder']/(row['name']+'.svg')).getroot()
        assert svg.get('viewBox') == '0 0 532.8 268.92'
        panels = svg.findall('.//s:clipPath/s:rect', typography.NS)
        dimensions = [[float(p.get('width')), float(p.get('height'))] for p in panels]
        assert len(dimensions) == 2
        vector_panels.append(dimensions)
    assert np.allclose(vector_panels, vector_panels[0])
    assert all(report.sha(ROOT/p) == digest for p, digest in before.items())
    assert all(report.sha(ROOT/p) == digest for p, digest in original_650.items())
    assert all(report.sha(DEST/p) == digest for p, digest in other_figures.items())
    audit = dict(passed=True, no_new_experiments=True, figure_inches=list(typography.SIZE),
        physical_panels_pt=vector_panels[0], algorithm_labels_on_one_line=True,
        minimum_category_gap_pt=6., category_font_pt=12, numeric_tick_font_pt=12,
        axis_label_font_pt=14, original_650kJ_artifacts_unchanged=original_650,
        unchanged_measurement_sha256=before, source_sha256=sources, updated=updated,
        paper_rank_alias_budget_kJ=300)
    report.write(BASE/'matched_rank_fu_layout_v1_checks.json', audit)

    limited_checks = report.read(limited/'verification.json')
    limited_checks['presentation_revision'] = 'matched_rank_fu_v1'
    limited_checks['source_sha256'].update(sources)
    limited_source = Path(__file__).with_name('limited_energy_rank_report_v1.py')
    limited_checks['source_sha256'][str(limited_source.relative_to(ROOT))] = report.sha(limited_source)
    report.write(limited/'verification.json', limited_checks)
    font_audit = report.read(BASE/'data_typography_v1.json')
    for row in font_audit['updated']:
        if row['name'] == BAR and row['folder'] == str(DEST.relative_to(ROOT)):
            row['typography'] = result['typography']
    font_audit['source_sha256'].update(sources)
    font_audit['matched_rank_fu_layout_checks'] = 'matched_rank_fu_layout_v1_checks.json'
    report.write(BASE/'data_typography_v1.json', font_audit)

    (target/'README.md').write_text(
        '# Matching paper figures\n\n'
        'Use every PDF with `width=\\linewidth`. All four have the same 532.8 pt canvas width, '
        '12 pt numeric ticks, and 14 pt axis titles and dataset labels.\n\n'
        '- `all_method_rank_boxplot_1x2`: all retained-query ranks at **300 kJ**, copied from '
        '`../downlink_bandwidth/limited_energy/300kJ/19_all_method_rank_boxplot_1x2.pdf`. '
        'See the accompanying caption for the experiment conditions.\n'
        '- `fu_steps_energy_1x2`: current Figure 11, paired 1-step and 9-step FU workloads.\n'
        '- `01_round_loss_1x2`: current P-KS round validation loss.\n'
        '- `05_time_mr_1x2`: current P-KS retained mR under the shared time budget.\n\n'
        'The rank and FU bar figures have identical 7.4 × 3.735 inch canvases and '
        '205.128 × 174.798 pt plotting panels. Both use single-line 12 pt algorithm names, '
        'with at least 6 pt between adjacent names. Both figures show all four panel borders. '
        'The FU legend remains 12 pt and 8 pt '
        'above the panels. All measured values and outliers are unchanged. The 200/400 kJ '
        'rank alternatives use the same layout; the historic 650 kJ export is unchanged.\n\n'
        'Reapply this focused presentation pass after rebuilding plots: '
        '`.venv/bin/python tools/uav_convergence_v2/match_rank_fu_layout_v1.py`.\n')
    print(dict(passed=True, updated_figures=3 if only_rank else len(updated), physical_panels_pt=vector_panels[0],
               measurement_files_unchanged=len(before)), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-rank', action='store_true')
    main(only_rank=parser.parse_args().only_rank)
