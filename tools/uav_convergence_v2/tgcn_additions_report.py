"""Energy/utility, FU workload and native text/video result illustrations."""
import copy
import html
import math
from pathlib import Path
import textwrap
import time

import numpy as np
import proposed_pks_report as current
import learning_time_report as continuation
import tgcn_fu_workload_v1 as fu
import tgcn_retrieval_examples_v1 as retrieval

ROOT, read, write, sha = current.ROOT, current.read, current.write, current.sha
DEST = ROOT/'final_final_charts/downlink_bandwidth'
prior, export, plots = current.prior, current.export, current.plots
METHODS, DATASETS = current.METHODS, current.DATASETS
NAMES = ['10_energy_mr_1x2', '11_fu_steps_energy_1x2',
         '12_capera_text_video_examples', '13_airscape_text_video_examples',
         '14_capera_energy_matched_retrieval', '15_airscape_energy_matched_retrieval']
ENERGY_BUDGET_KJ = 650.


def setup():
    export.setup(DEST, METHODS, current.source)
    for key, index in [('COLORS', 0), ('MARKERS', 1), ('STYLES', 2)]:
        setattr(prior.chart, key, [p[index] for p in current.PALETTE])
        setattr(prior.chart.style, key, [p[index] for p in current.PALETTE])


def energy_utility():
    paths, checks, targets, budget = {}, [], [], []
    for ds in DATASETS:
        for method, label in METHODS.items():
            path, _, check = continuation.trajectory(ds, method)
            assert path[-1]['energy_j'] > ENERGY_BUDGET_KJ*1000
            points = [p for p in path if p['energy_j'] <= ENERGY_BUDGET_KJ*1000]
            assert points[-1]['FU_completed'] == 45
            assert all(b['energy_j'] > a['energy_j'] for a, b in zip(points, points[1:]))
            paths[(ds, method)] = points
            checks.append(check)
            for point in points:
                budget.append(dict(dataset=ds, method=method, energy_kJ=point['energy_j']*.001,
                    active_s=point['active_s'], FL_round=point['FL_round'], FU_completed=point['FU_completed'],
                    retained_mR_percent=100*point['evaluation']['retained']['mR'], stage=point['stage']))
            for target in ([18., 20., 22.] if ds == 'CapERA' else [14., 15., 16.]):
                first = next((p for p in points if p['FU_completed'] == 45 and 100*p['evaluation']['retained']['mR'] >= target), None)
                targets.append(dict(dataset=ds, method=method, label=label, target_mR_percent=target,
                    reached=first is not None, energy_kJ=None if first is None else first['energy_j']*.001,
                    active_s=None if first is None else first['active_s'], all_FU_procedures_required=True))
    fig, axes = prior.chart.style.canvas('Retained mR (%)')
    panels = []
    ticks = np.linspace(0, ENERGY_BUDGET_KJ, 6).tolist()
    for ax, ds in zip(axes, DATASETS):
        values = []
        for i, (method, title) in enumerate(METHODS.items()):
            points = paths[(ds, method)]
            x = [p['energy_j']*.001 for p in points]
            y = [100*p['evaluation']['retained']['mR'] for p in points]
            dx, dy = x+[ENERGY_BUDGET_KJ], y+[y[-1]]
            line = prior.chart.style.curve(ax, dx, dy, i, title)
            line.set_drawstyle('steps-post')
            line.set_markevery([])
            # Mark the same energy budgets using the last completed evaluated model.
            my = [y[max(j for j, v in enumerate(x) if v <= tick)] for tick in ticks]
            ax.plot(ticks, my, linestyle='None', marker=current.PALETTE[i][1],
                    color=current.PALETTE[i][0], markerfacecolor='white', markersize=line.get_markersize(),
                    markeredgewidth=line.get_markeredgewidth(), label='_nolegend_')
            values += y
            prior.SERIES.append(dict(figure=NAMES[0], dataset=ds, method=method, metric='retained_mR_percent',
                x=x, y=y, display_x=dx, display_y=dy, marker_x=ticks, marker_y=my,
                drawstyle='steps-post', last_completed_evaluated_model=True, no_interpolation=True,
                all_FU_completed=[p['FU_completed'] == 45 for p in points]))
        prior.chart.regular_axis(ax, 'x', ticks, 0)
        ax.set_xlabel('Cumulative energy (kJ)')
        panels.append(values)
    prior.chart.finish_axes(axes, panels, decimals=1)
    prior.save(fig, axes, NAMES[0], 'Retained text/video retrieval performance versus cumulative energy')
    write(DEST/'data/energy_utility_checkpoints.json', budget)
    prior.write_csv(DEST/'data/energy_utility_checkpoints.csv', budget)
    write(DEST/'data/energy_to_mr_targets.json', targets)
    prior.write_csv(DEST/'data/energy_to_mr_targets.csv', targets)
    return dict(passed=True, trajectories=checks, energy_budget_kJ=ENERGY_BUDGET_KJ,
                all_methods_finish_all45_FU_procedures_within_budget=True,
                warm20_before_energy_origin=True, extrapolated_model_updates=False)


def fu_metrics():
    assert read(fu.OUT/'campaign_finished.json')['passed']
    assert read(fu.OUT/'control_checks.json')['passed']
    old_source = prior.source
    prior.source = lambda case, ds, method: fu.source(int(case.rsplit('_', 1)[-1]), ds, method)
    rows, checks = [], []
    hashes = {}
    def digest(path):
        key = str(path.resolve())
        if key not in hashes:
            hashes[key] = sha(path)
        prior.SOURCES[str(path.relative_to(ROOT))] = hashes[key]
        return hashes[key]
    try:
        for value in fu.VALUES:
            for ds in DATASETS:
                anchor = current.source('baseline', ds, 'gate_select16')
                for method in METHODS:
                    arm = fu.source(value, ds, method)
                    for name in ['plan.json', 'initial20/final.pt', 'feature_cache/features.pt', 'reference200/final.pt']:
                        assert digest(arm.parent/name) == digest(anchor.parent/name)
                    settings = prior.load(arm.parents[1]/'settings.json')
                    old = prior.load(anchor.parents[1]/'settings.json')
                    assert settings['config'] == old['config'] and settings['energy_model'] == old['energy_model']
                    for name in ['deployment.json', 'mission_spec.json']:
                        assert digest(arm.parents[1]/name) == digest(anchor.parents[1]/name)
                    requests = copy.deepcopy(prior.load(arm.parent/'requests.json'))
                    assert len(requests) == 45 and all(r['required_steps'] == value for r in requests)
                    for r in requests:
                        r['required_steps'] = 5
                    assert requests == prior.load(anchor.parent/'requests.json')
                    row = prior.ledger(fu.case_id(value), ds, method)
                    check = prior.load(arm/'verification.json')
                    assert check['FL_rounds'] == 200 and check['FU_steps'] == row['FU_cycles'] == value*45
                    tx = [prior.load(p) for p in sorted((arm/'transactions').glob('*.json'))]
                    row.update(fu_steps_per_request=value, FL_rounds=200, FU_requests=45,
                        origin='fresh_execution',
                        joint_transactions=sum(t['kind'] == 'joint' for t in tx),
                        FL_only_transactions=sum(t['kind'] == 'fl' for t in tx),
                        FU_only_transactions=sum(t['kind'] == 'fu' for t in tx))
                    assert sum(t['kind'] in ['fl', 'joint'] for t in tx) == 180
                    assert sum(len(t['FU_request_ids']) for t in tx) == value*45
                    if value == 5:
                        old_row = next(r for r in read(DEST/'data/metrics.json') if (r['dataset'], r['method'], r['bandwidth_mhz']) == (ds, method, 1))
                        for key in ['total_energy_j', 'delay_s', 'retained_mR_percent', 'retained_loss', 'forget_KL']:
                            assert math.isclose(row[key], old_row[key], rel_tol=1e-12, abs_tol=1e-8)
                    rows.append(row)
                    checks.append(dict(dataset=ds, method=method, steps=value, shared_scenario_and_models=True,
                        source=row['source'], actual_complete_workload=True, costs_recomputed=True))
    finally:
        prior.source = old_source
    fig, axes = prior.chart.style.canvas('Total energy (kJ)')
    panels = []
    for ax, ds in zip(axes, DATASETS):
        values = []
        for i, (method, title) in enumerate(METHODS.items()):
            y = [next(r['total_energy_j']*.001 for r in rows if (r['dataset'], r['method'], r['fu_steps_per_request']) == (ds, method, v)) for v in fu.VALUES]
            prior.chart.style.curve(ax, fu.VALUES, y, i, title).set_markevery(1)
            values += y
            prior.SERIES.append(dict(figure=NAMES[1], dataset=ds, method=method, x=fu.VALUES, y=y,
                                     metric='total_energy_j', scale=.001, axis='fu_steps'))
        prior.chart.regular_axis(ax, 'x', fu.VALUES, 0)
        ax.set_xlabel('FU steps per request')
        panels.append(values)
    prior.chart.finish_axes(axes, panels)
    prior.save(fig, axes, NAMES[1], 'Total energy versus FU work per deletion request')
    write(DEST/'data/fu_workload_metrics.json', rows)
    prior.write_csv(DEST/'data/fu_workload_metrics.csv', rows)
    write(DEST/'data/fu_workload_protocol.json', fu.register())
    write(DEST/'data/fu_workload_controls.json', read(fu.OUT/'control_checks.json'))
    return dict(passed=True, new_arms=40, zero_change_controls=8, cases=checks)


def retrieval_figures(study=retrieval, names=None, prefix='retrieval'):
    from PIL import Image
    assert read(study.OUT/'finished.json')['passed']
    examples = read(study.OUT/'examples.json')
    frame_cache, provenance = {}, []
    # Frames are decoded from original local videos; no image synthesis or retouching.
    for ds, name in zip(DATASETS, names or NAMES[2:4]):
        data, _ = study.data_for(ds)
        cases = [c for c in examples if c['dataset'] == ds and c['available']]
        assert len(cases) == 2, 'Do not fabricate a missing contrasting example.'
        fig = prior.chart.plt.figure(figsize=(10.4, 5.7), facecolor='white')
        condition = '650 kJ energy budget' if prefix != 'retrieval' else '200 FL rounds'
        fig.text(.5, .995, ds+' | '+condition, ha='center', va='top', fontsize=13)
        layout = []
        for row, case in enumerate(cases):
            top = .935-row*.48
            query = textwrap.fill(' '.join(case['query'].split()), width=112)
            query_lines = query.splitlines()
            display_query = '\n'.join(query_lines[:3])
            if len(query_lines) > 3:
                display_query = display_query.rstrip()+' …'
            fig.text(.025, top, f'({chr(97+row)}) '+display_query, ha='left', va='top', fontsize=12)
            columns = [('Ground truth', case['ground_truth_video'], None)]
            columns += [(label, case[m+'_top1_video'], m) for m, label in METHODS.items()]
            image_y = top-.34
            for col, (label, vid, method) in enumerate(columns):
                key = (ds, vid)
                if key not in frame_cache:
                    frames, indices = data.frames(vid)
                    frame = frames[3].copy()
                    frame.thumbnail((480, 300), Image.Resampling.LANCZOS)
                    frame_cache[key] = frame
                    provenance.append(dict(dataset=ds, video_id=vid, file=str(data.media[vid].relative_to(ROOT)),
                        sha256=sha(data.media[vid]), frame_index=indices[3], training_feature_frame_indices=indices,
                        representative_frame_ordinal=3, thumbnail_size=list(frame.size), altered_content=False))
                left, width = .025+col*.195, .178
                ax = fig.add_axes([left, image_y, width, .20])
                ax.imshow(frame_cache[key]);ax.set_xticks([]);ax.set_yticks([])
                hit = vid == case['ground_truth_video']
                edge = '#333333' if method is None else '#438F56' if hit else '#C65365'
                for spine in ax.spines.values():
                    spine.set_color(edge);spine.set_linewidth(1.8)
                fig.text(left+width/2, image_y+.218, label, ha='center', va='bottom', fontsize=12)
                rank = None if method is None else case[method+'_rank']
                text = f'Video {vid}' if method is None else f'GT rank: {rank}'
                fig.text(left+width/2, image_y-.022, text, ha='center', va='top', fontsize=11)
                layout.append(dict(category=case['category'], caption_id=case['caption_id'], method=method,
                    video_id=vid, ground_truth_video=case['ground_truth_video'], GT_rank=rank, top1_correct=hit,
                    border_color=edge))
            case['display_query'] = display_query
        files = []
        for ext in ['png', 'pdf', 'svg']:
            p = DEST/(name+'.'+ext)
            fig.savefig(p, dpi=350, bbox_inches='tight')
            files.append(dict(file=p.name, sha256=sha(p)))
        prior.chart.plt.close(fig)
        prior.FIGURES.append(dict(name=name, title=ds+(' text-to-video retrieval at650kJ' if prefix != 'retrieval' else ' text-to-video retrieval at200 rounds'),
            kind='retrieval_examples', axes=[], files=files, layout=layout))
    write(DEST/'data'/(prefix+'_examples.json'), examples)
    write(DEST/'data'/(prefix+'_frame_provenance.json'), provenance)
    write(DEST/'data'/(prefix+'_examples_protocol.json'), study.register())
    summary = read(study.OUT/'summary.json')
    write(DEST/'data'/(prefix+'_summary.json'), summary)
    prior.write_csv(DEST/'data'/(prefix+'_summary.csv'), summary)
    for ds in DATASETS:
        for kind in ['query_ranks', 'paired_counts']:
            data = read(study.OUT/(ds.lower()+'_'+kind+'.json'))
            write(DEST/'data'/(prefix+'_'+ds.lower()+'_'+kind+'.json'), data)
            prior.write_csv(DEST/'data'/(prefix+'_'+ds.lower()+'_'+kind+'.csv'), data)
    return dict(passed=True, verified_models=8, original_recall_and_loss_reproduced=True,
                actual_video_frames=len(provenance), selected_examples=4, gain_and_loss_examples=True,
                generated_captions_or_media=False, all_retained_query_ranks_published=True)


def documents():
    lines = [
        '# Energy efficiency and actual video–text retrieval', '',
        '[All figures](index.html) · [FU workload metrics](data/fu_workload_metrics.csv) · [Retrieval metrics](data/retrieval_summary.csv)', '',
        'Current Proposed is unchanged P-KS. All other algorithms, models and data definitions retain their published identities. No model is selected or retrained to manufacture illustrative differences.', '',
        '## 10: Retained mR versus cumulative energy', '',
        'Uses the existing actual completed-model trajectories, with communication, executed adapter computation and active hovering independently reconstructed from transaction ledgers. All methods receive the same650kJ active-work budget. Zero is the common warm20 checkpoint; preflight feature extraction, warmup, travel and IoT collection are excluded consistently with the existing resource figures. No energy is inferred merely from elapsed time.', '',
        'All45 deletion procedures finish by650kJ for every method. Before completion, points reflect concurrent FL/FU progress; they do not imply that all requests have already finished. After the original200FL/225FU workload, every method continues retained-only FL with the same restarted0.0035 learning rate. No new deletion requests arrive. Curves show the last completed evaluated model using steps-post, with markers at the same energy budgets. No interpolation or unexecuted model updates. All recorded declines are retained.', '',
        'Energy-to-target CSV requires all45 FU procedures finished; missing targets remain unreached. Procedure completion and retained mR do not certify forgetting quality. Existing P-KS common-retrain KL limitations remain; this figure is an energy/retained-utility comparison, not an equal-certified-unlearning claim.', '',
        '## 11: Total energy versus FU steps per request', '',
        'Forty fresh executions:1/3/5/7/9 outer FU steps per request × four methods × two datasets. Same45 requests, sample cohorts, release ordinals,10UAV, five regions,200 totalFL rounds, initial models, features, learning/radio/energy coefficients. Every prescribed FU step finishes. Eight fresh five-step controls exactly reproduce original final parameter tensors, quality, transaction clocks and radio decisions.', '',
        'GA→FT comparators retain two local optimizer substeps per outer FU cycle; both are charged. More steps do not necessarily improve mR or KL, so all quality values and physical-bound exceedances accompany the energy figure. Flat Proposed curves are retained: co-aggregation may absorb additional FU work within the unchanged180 mission-FL transmissions. Parallel refers to joint FL/FU aggregation/application, not simultaneous local CPU execution.', '',
        '## 12–13: Native text-to-video retrieval examples', '',
        'Evaluate the unchanged final200 checkpoints with all45 requests/225FU cycles complete, at the same workload rather than the same energy. Native captions, original gallery and cached eight-frame RemoteCLIP features are unchanged. All six recalls and validation loss are independently reproduced before selecting illustrations. A displayed image is the fourth of the same eight original sampled video frames, downsampled only to fit the figure. It is not a generated image and the task is video retrieval, not caption generation.', '',
        'Each dataset shows one selected Proposed rank advantage and one disadvantage. Prefer a top1 reversal among unique native caption strings; choose the median rank gap with caption-ID tie breaking. Full selection rules and counts are in the protocol. These two deliberately contrasting cases are not a random or representative sample. Complete retained-query rank/top3-ID tables and better/equal/worse counts prevent a favorable-only interpretation.', '',
        'Ground truth is the annotated paired video, not a judgment that every other semantically similar video is irrelevant. Green borders indicate the correct top1 video, red an incorrect top1; GT rank gives the annotated video position in the full gallery. Full native query text and media/frame hashes accompany selected cases. Long displayed queries may be explicitly abbreviated, but inference always used the original complete caption. Only derived representative-frame figures are published; raw videos/features/checkpoints stay local.', '',
        '## 14–15: Actual retrieval under the same650kJ energy budget', '',
        'These use the exact last completed model within650kJ from the energy/mR curve, rather than the common200-round checkpoint. All45 FU procedures complete. The original continuation was replayed unchanged; every evaluation/energy/time point and final parameter tensor reproduced exactly before intermediate checkpoints were used. Different completed FL counts are consequences of the actual energy costs, not extra uncharged updates. The same illustration selection rule and full-query reporting apply. The unchanged200-round examples remain above as a separate comparison.', '',
        'Single development seed and frozen-feature rank10 adapter study, not a full-backbone VLM hardware benchmark or statistical superiority result.', '',
        '## Reproduction', '', '```sh',
        '.venv/bin/python tools/uav_convergence_v2/tgcn_fu_workload_v1.py run',
        '.venv/bin/python tools/uav_convergence_v2/tgcn_retrieval_examples_v1.py run',
        '.venv/bin/python tools/uav_convergence_v2/tgcn_energy_checkpoint_v1.py',
        '.venv/bin/python tools/uav_convergence_v2/tgcn_energy_retrieval_v1.py',
        '.venv/bin/python tools/uav_convergence_v2/tgcn_additions_report.py',
        '.venv/bin/python tools/uav_convergence_v2/tgcn_additions_check.py', '```', '']
    (DEST/'TGCN_ADDITIONS.md').write_text('\n'.join(lines))
    marker = '<!-- tgcn-additions -->'
    end = '<!-- /tgcn-additions -->'
    for folder in [DEST, DEST.parent, DEST.parent/'proposed_pks']:
        relative = '' if folder == DEST else 'downlink_bandwidth/' if folder == DEST.parent else '../downlink_bandwidth/'
        path = folder/'README.md'
        text = path.read_text()
        if marker in text:
            a, b = text.index(marker), text.index(end)+len(end)
            text = text[:a]+text[b:]
        insert = text.index('\n')+1
        card = ('\n'+marker+'\n[Energy–mR, FU workload and actual text/video retrieval examples]('+relative+'TGCN_ADDITIONS.md). Figures10–15; unchanged Proposed=P-KS.\n'+end+'\n')
        path.write_text(text[:insert]+card+text[insert:])
        path = folder/'index.html'
        text = path.read_text()
        if marker in text:
            a, b = text.index(marker), text.index(end)+len(end)
            text = text[:a]+text[b:]
        insert = text.index('</h1>')+len('</h1>')
        figures = prior.FIGURES if folder == DEST else [f for f in prior.FIGURES if f['name'] in [NAMES[0], NAMES[4]]]
        cards = ''.join('<section><h2>'+html.escape(f['title'])+'</h2><a href="'+relative+f['name']+'.pdf"><img loading="lazy" src="'+relative+f['name']+'.png"></a></section>' for f in figures)
        text = text[:insert]+marker+'<p><a href="'+relative+'TGCN_ADDITIONS.md">Energy efficiency and actual VLM retrieval: definitions and full numerical results</a></p>'+cards+end+text[insert:]
        path.write_text(text)


def main():
    original = read(DEST/'data/manifest.json')
    # Keep this extension's metadata separate so all nine original figures and
    # their audited measurements remain byte-for-byte unchanged.
    old_hashes = {f['file']: sha(DEST/f['file']) for figure in original['figures'] for f in figure['files']}
    setup()
    audits = dict(energy_utility=energy_utility(), fu_workload=fu_metrics(), retrieval=retrieval_figures())
    import tgcn_energy_retrieval_v1 as energy_retrieval
    audits['energy_matched_retrieval'] = retrieval_figures(energy_retrieval, NAMES[4:6], 'retrieval_energy')
    write(DEST/'data/tgcn_audit.json', dict(passed=True, **audits))
    write(DEST/'data/tgcn_plot_series.json', prior.SERIES)
    for module in [Path(__file__), Path(fu.__file__), Path(retrieval.__file__)]:
        prior.SOURCES[str(module.relative_to(ROOT))] = sha(module)
    manifest = dict(complete=True, unix=time.time(), methods=METHODS, figures=prior.FIGURES,
        source_sha256=prior.SOURCES, fresh_FU_workload_arms=40, original_figures=old_hashes,
        actual_retrieval_models=16, scope='Scientific figure extension; base manifest/metrics/artifacts unchanged.')
    write(DEST/'data/tgcn_manifest.json', manifest)
    assert all(sha(DEST/p) == digest for p, digest in old_hashes.items())
    documents()
    print(dict(passed=True, new_figures=len(prior.FIGURES), new_training_arms=40, destination=str(DEST)), flush=True)


if __name__ == '__main__':
    main()
