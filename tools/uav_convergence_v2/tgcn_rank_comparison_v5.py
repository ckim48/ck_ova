"""Full-query paired outcomes and four-method GT-rank distributions at 650 kJ."""
from pathlib import Path

import numpy as np
from matplotlib.cbook import boxplot_stats
from matplotlib.patches import Patch


def render(report):
    prior, dest = report.prior, report.DEST
    methods = report.METHODS
    comparators = [m for m in methods if m != 'gate_select16']
    labels = ['Improved', 'Tied', 'Worsened']
    colors, hatches = ['#4C72B0', '#E6B45A', '#C65365'], ['//', '', '..']
    sources, ranks, paired, records, boxes = {}, {}, {}, [], []
    for ds in report.DATASETS:
        for suffix in ['query_ranks', 'paired_counts']:
            path = dest/'data'/f'retrieval_energy_{ds.lower()}_{suffix}.json'
            sources[str(path.relative_to(report.ROOT))] = report.sha(path)
            (ranks if suffix == 'query_ranks' else paired)[ds] = report.read(path)
    sources[str(Path(__file__).relative_to(report.ROOT))] = report.sha(Path(__file__))
    prior.SOURCES.update(sources)
    prior.chart.plt.rcParams['hatch.linewidth'] = .55

    def save(fig, name, title, axes, **metadata):
        files = []
        for ext in ['pdf', 'png', 'svg']:
            path = dest/(name+'.'+ext)
            fig.savefig(path, dpi=350, bbox_inches='tight')
            files.append(dict(file=path.name, sha256=report.sha(path)))
        prior.chart.plt.close(fig)
        prior.FIGURES.append(dict(name=name, title=title, axes=axes, files=files, **metadata))

    def panel_label(fig, ax, ds):
        pos = ax.get_position()
        fig.text((pos.x0+pos.x1)/2, .015,
                 f'({chr(97+report.DATASETS.index(ds))}) {ds}',
                 ha='center', va='bottom', fontsize=14)

    fig, axes = prior.chart.style.canvas('Share of queries (%)')
    panels, artists = [], []
    ticks = [0, 20, 40, 60, 80, 100]
    for ax, ds in zip(axes, report.DATASETS):
        for i, method in enumerate(comparators):
            delta = [r[method+'_rank']-r['gate_select16_rank'] for r in ranks[ds]]
            counts = [sum(d > 0 for d in delta), sum(d == 0 for d in delta), sum(d < 0 for d in delta)]
            known = next(p for p in paired[ds] if p['comparator'] == method)
            assert counts == [known[k] for k in ['better', 'equal', 'worse']]
            assert sum(counts) == len(ranks[ds])
            bottom = 0.
            for label, color, hatch, count in zip(labels, colors, hatches, counts):
                percent = 100*count/len(ranks[ds])
                bar = ax.bar(i, percent, bottom=bottom, width=.62, color=color,
                             edgecolor='#202020', linewidth=.65, hatch=hatch)[0]
                assert bar.get_height() == percent and bar.get_y() == bottom
                records.append(dict(dataset=ds, comparator=method, comparator_label=methods[method],
                    outcome=label.lower(), count=count, queries=len(ranks[ds]), percent=percent,
                    energy_budget_kJ=650., reference_method='gate_select16'))
                artists.append(dict(dataset=ds, comparator=method, outcome=label.lower(),
                    x=i, bottom=bottom, height=percent, color=color, hatch=hatch))
                bottom += percent
            assert abs(bottom-100.) < 1e-10
        ax.set_xlim(-.55, len(comparators)-.45)
        ax.set_xticks(range(len(comparators)), [methods[m] for m in comparators])
        ax.tick_params(axis='x', labelsize=9, length=0, pad=6)
        ax.set_xlabel('Comparator', fontsize=11)
        ax.grid(False, axis='x')
        prior.chart.regular_axis(ax, 'y', ticks, 0)
        assert not ax.texts and len(ax.patches) == 9
        panel_label(fig, ax, ds)
        panels.append(dict(dataset=ds, x=dict(kind='categorical', ticks=list(range(3)),
            limits=list(ax.get_xlim()), labels=[methods[m] for m in comparators]),
            y=dict(ticks=ticks, limits=[0, 100], labels=[str(v) for v in ticks])))
    fig.legend(handles=[Patch(facecolor=c, edgecolor='#202020', linewidth=.65, hatch=h, label=label)
                        for label, c, h in zip(labels, colors, hatches)],
               title='Proposed vs. each comparator at 650 kJ', title_fontsize=10,
               ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.04), fontsize=10,
               handlelength=1.7, columnspacing=1.5, frameon=False)
    save(fig, report.NAMES[8], 'All-query rank outcomes: Proposed versus all comparators', panels,
         kind='paired_outcome_bars', bars=artists, energy_budget_kJ=650.,
         comparator_order=comparators, all_retained_queries=True, numeric_bar_annotations=False,
         caption='Proposed GT-rank outcomes relative to each comparator over all retained text queries at the same 650 kJ active-energy budget. Improved, tied and worsened mean smaller, equal and larger paired-video rank, respectively. Each comparator bar sums to 100%. These are paired rank outcomes, not Recall@1, mR or independent-seed uncertainty. Proposed is the reference, so a self-comparison bar is omitted.')

    fig, axes = prior.chart.style.canvas('GT rank')
    panels = []
    # A common linear scale contains every observed rank, including all outliers.
    maximum = max(r[m+'_rank'] for ds in report.DATASETS for r in ranks[ds] for m in methods)
    upper = 100*(int(maximum//100)+1)
    ticks = list(range(0, upper+1, 100))
    for ax, ds in zip(axes, report.DATASETS):
        for i, (method, label) in enumerate(methods.items()):
            values = np.array([r[method+'_rank'] for r in ranks[ds]], dtype=float)
            stats = boxplot_stats(values, whis=1.5)[0]
            color = report.current.PALETTE[i][0]
            made = ax.bxp([stats], positions=[i], widths=.52, patch_artist=True,
                showfliers=True, showmeans=False, manage_ticks=False,
                boxprops=dict(facecolor=color, edgecolor='#202020', linewidth=.8, alpha=.8),
                medianprops=dict(color='#202020', linewidth=1.2),
                whiskerprops=dict(color='#202020', linewidth=.75),
                capprops=dict(color='#202020', linewidth=.75),
                flierprops=dict(marker='o', markersize=1.8, markerfacecolor=color,
                                markeredgecolor=color, markeredgewidth=.2, alpha=.55))
            assert len(made['fliers'][0].get_ydata()) == len(stats['fliers'])
            assert max(values) < upper and min(values) >= 1
            boxes.append(dict(dataset=ds, method=method, label=label, queries=len(values),
                energy_budget_kJ=650., minimum=float(min(values)), maximum=float(max(values)),
                q1=float(stats['q1']), median=float(stats['med']), q3=float(stats['q3']),
                whisker_low=float(stats['whislo']), whisker_high=float(stats['whishi']),
                fliers=stats['fliers'].tolist(), color=color))
        ax.set_xlim(-.55, len(methods)-.45)
        ax.set_xticks(range(len(methods)), list(methods.values()))
        ax.tick_params(axis='x', labelsize=8.5, length=0, pad=6)
        ax.grid(False, axis='x')
        prior.chart.regular_axis(ax, 'y', ticks, 0)
        assert not ax.texts
        panel_label(fig, ax, ds)
        panels.append(dict(dataset=ds, x=dict(kind='categorical', ticks=list(range(4)),
            limits=list(ax.get_xlim()), labels=list(methods.values())),
            y=dict(ticks=ticks, limits=[0, upper], labels=[str(v) for v in ticks])))
    fig.suptitle('Ground-truth video rank at 650 kJ (lower is better)', fontsize=11, y=.98)
    save(fig, report.NAMES[9], 'All-method full-query GT-rank boxplots', panels,
         kind='rank_boxplot', boxes=boxes, energy_budget_kJ=650., all_retained_queries=True,
         whisker_rule='1.5 IQR, ending at the most extreme observed in-fence values',
         show_all_fliers=True, y_scale='linear',
         caption='Distribution of annotated video ranks for all retained text queries under the same 650 kJ active-energy budget. Boxes show the 25th–75th percentiles with median lines; whiskers use 1.5 IQR and every outside value is plotted. Lower ranks are better. This is the query distribution from one development seed, not training-run confidence intervals. CapERA captions from the same video are not independent observations.')
    report.write(dest/'data/retrieval_all_comparator_outcomes.json', records)
    prior.write_csv(dest/'data/retrieval_all_comparator_outcomes.csv', records)
    report.write(dest/'data/retrieval_rank_box_stats.json', boxes)
    prior.write_csv(dest/'data/retrieval_rank_box_stats.csv', boxes)
    report.write(dest/'data/retrieval_rank_comparison_protocol.json', dict(
        version='all_methods_v5', display_only=True, energy_budget_kJ=650., source_sha256=sources,
        query_counts={ds: len(ranks[ds]) for ds in report.DATASETS},
        comparison='Paired ranks of the annotated video for every retained native text query. All three baselines compared with Proposed; a separate raw-rank boxplot includes all four algorithms.',
        selection='No query filtering, wins-only selection, outlier suppression, smoothing, retraining or score changes.',
        uncertainty='Single development seed. Boxes describe query ranks, not independent-run uncertainty or statistical significance. Multiple CapERA captions may share a video.',
        models='Last completed checkpoints within the common 650 kJ budget, with all 45 FU procedures complete; FL counts may differ. This retained-retrieval view does not certify forgetting.'))
    return dict(passed=True, all_methods=4, comparator_pairs=6, boxplots=8,
                retained_query_counts=[1280, 256], no_query_filtering=True,
                all_outliers_visible=True, models_and_scores_unchanged=True)
