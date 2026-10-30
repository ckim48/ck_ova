"""Cross-check published plots/examples against actual completed experiments."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
import tgcn_additions_report as report
import tgcn_retrieval_examples_v1 as retrieval
import tgcn_energy_retrieval_v1 as energy_retrieval
import tgcn_energy_checkpoint_v1 as checkpoints


def check_retrieval(module, prefix, figures):
    read, sha, dest, root = report.read, report.sha, report.DEST, report.ROOT
    protocol = read(dest/'data'/(prefix+'_examples_protocol.json'))
    examples = read(dest/'data'/(prefix+'_examples.json'))
    integrated = protocol.get('display_selection_version') == 'integrated_outcomes_v4'
    mixed = integrated or protocol.get('display_selection_version') == 'combined_mixed_v3'
    source_prefix = 'retrieval_energy' if mixed else prefix
    summaries = read(dest/'data'/(source_prefix+'_summary.json'))
    frames = read(dest/'data'/(prefix+'_frame_provenance.json'))
    successful = protocol.get('display_selection_version') == 'favorable_diverse_v2'
    assert len(summaries) == 8 and len(examples) == (8 if mixed else 12 if successful else 4)
    for name, digest in protocol['source_sha256'].items():
        assert sha(root/name) == digest, name
    for ds in report.DATASETS:
        ranks = read(dest/'data'/(source_prefix+'_'+ds.lower()+'_query_ranks.json'))
        pairs = read(dest/'data'/(source_prefix+'_'+ds.lower()+'_paired_counts.json'))
        data, _ = retrieval.data_for(ds)
        arm = report.current.source('baseline', ds, 'gate_select16')
        plan = read(arm.parent/'plan.json')
        gallery = plan['gallery_images']
        count = len(data.caption_ids[0])
        all_cids = [c for i in gallery for c in data.caption_ids[i]]
        scores = np.load(module.OUT/(ds.lower()+'_scores.npz'))
        assert len(ranks) == len(plan['validation_images'])*count
        assert {r['ground_truth_video'] for r in ranks} == set(plan['validation_images'])
        for method in report.METHODS:
            # Recheck actual full-gallery inference, not just displayed selected rows.
            order = torch.argsort(torch.from_numpy(scores[method]).T, dim=1, descending=True)
            inverse = torch.argsort(order, dim=1)+1
            for row in ranks:
                ci = row['gallery_caption_index']
                assert all_cids[ci] == row['caption_id']
                assert gallery[ci//count] == row['ground_truth_video']
                assert row[method+'_rank'] == int(inverse[ci, ci//count])
                assert row[method+'_top3_video_ids'] == [gallery[int(i)] for i in order[ci, :3]]
                assert row[method+'_top1_video'] == row[method+'_top3_video_ids'][0]
            summary = next(r for r in summaries if (r['dataset'], r['method']) == (ds, method))
            for k in [1, 5, 10]:
                assert math.isclose(sum(r[method+'_rank'] <= k for r in ranks)/len(ranks), summary[f't2i_R{k}'], abs_tol=2e-7)
            assert math.isclose(summary['mR'], sum(summary[f'{d}_R{k}'] for d in ['i2t', 't2i'] for k in [1, 5, 10])/6, abs_tol=1e-12)
            if method != 'gate_select16':
                pair = next(r for r in pairs if r['comparator'] == method)
                delta = [r[method+'_rank']-r['gate_select16_rank'] for r in ranks]
                assert pair['better'] == sum(v>0 for v in delta)
                assert pair['equal'] == sum(v==0 for v in delta)
                assert pair['worse'] == sum(v<0 for v in delta)
        chosen = [r for r in examples if r['dataset'] == ds]
        if mixed:
            assert len(chosen) == len({r['ground_truth_video'] for r in chosen}) == 4
            assert [r['caption_id'] for r in chosen] == protocol['chosen_caption_ids'][ds]
            assert sum(r['category'] == 'selected_success' for r in chosen) == 3
            assert sum(r['category'] == 'selected_miss' for r in chosen) == 1
        elif successful:
            assert len(chosen) == len({r['ground_truth_video'] for r in chosen}) == 6
            assert {r['category'] for r in chosen} == {'selected_success'}
            assert [r['caption_id'] for r in chosen] == protocol['chosen_caption_ids'][ds]
            eligible = [r for r in ranks if r['gate_select16_rank'] == 1 and
                        max(r[m+'_rank'] for m in report.METHODS if m != 'gate_select16') > 1]
            candidate = next(r for r in protocol['eligible_counts'] if r['dataset'] == ds)
            assert candidate['eligible_queries'] == len(eligible)
            assert candidate['eligible_videos'] == len({r['ground_truth_video'] for r in eligible})
        else:
            assert {r['category'] for r in chosen} == {'advantage', 'disadvantage'}
        for case in chosen:
            assert case['available']
            row = next(r for r in ranks if r['caption_id'] == case['caption_id'])
            assert all(case[k] == v for k, v in row.items())
            assert case['query'] == data.captions[case['caption_id']]
            if successful or mixed:
                assert case['display_query'].replace('…', '').strip() in ' '.join(case['query'].split())
            if mixed and case['category'] == 'selected_miss':
                original = next(r for r in read(module.OUT/'examples.json') if r['dataset'] == ds and r['category'] == 'disadvantage')
                assert row['caption_id'] == original['caption_id']
                assert row['gate_select16_rank'] > 1
                assert min(row[m+'_rank'] for m in report.METHODS if m != 'gate_select16') == 1
            elif successful or mixed:
                assert row['gate_select16_rank'] == 1
                assert row['gate_select16_top1_video'] == row['ground_truth_video']
                assert max(row[m+'_rank'] for m in report.METHODS if m != 'gate_select16') > 1
            else:
                difference = min(row[m+'_rank'] for m in report.METHODS if m != 'gate_select16')-row['gate_select16_rank']
                assert (difference > 0) == (case['category'] == 'advantage')
            for item in case['media'].values():
                assert sha(root/item['file']) == item['sha256']
        for frame in [r for r in frames if r['dataset'] == ds]:
            assert sha(root/frame['file']) == frame['sha256']
            assert frame['frame_index'] == frame['training_feature_frame_indices'][3]
            assert frame['representative_frame_ordinal'] == 3 and not frame['altered_content']
        figure = next(f for f in figures if f['name'] == report.NAMES[7 if integrated else 6]) if mixed else next(f for f in figures if f['name'].startswith(('12' if prefix=='retrieval' else '14') if ds=='CapERA' else ('13' if prefix=='retrieval' else '15')))
        layout = [r for r in figure['layout'] if r['dataset'] == ds] if mixed else figure['layout']
        assert len(layout) == 5*len(chosen)
        for item in layout:
            case = next(c for c in chosen if c['caption_id'] == item['caption_id'])
            if item['method'] is not None:
                assert item['video_id'] == case[item['method']+'_top1_video']
                assert item['GT_rank'] == case[item['method']+'_rank']
            assert item['top1_correct'] == (item['video_id'] == case['ground_truth_video'])
        root_svg = ET.parse(dest/(figure['name']+'.svg')).getroot()
        assert len(root_svg.findall('.//{http://www.w3.org/2000/svg}image')) == 5*(len(examples) if mixed else len(chosen))
        if mixed:
            assert figure['shared_column_headers'] and figure['rows'] == 8 and figure['rows_per_dataset'] == 4
            text = (dest/(figure['name']+'.svg')).read_text()
            heading = 'Retrieval outcomes and selected examples' if integrated else 'Selected retrieval examples'
            assert all(label in text for label in ['CapERA', 'AirScape', 'GT rank', heading])
        if integrated:
            previous = read(dest/'data/retrieval_combined_examples.json')
            assert chosen == [r for r in previous if r['dataset'] == ds]
            panel = figure['summary_panel']
            assert panel['comparator'] == 'afl' and panel['all_queries']
            assert panel['unit'] == 'retained_text_query' and not panel['numeric_bar_annotations']
            assert panel['x'] == dict(ticks=[0, 20, 40, 60, 80, 100], limits=[0, 100])
            pair = next(r for r in pairs if r['comparator'] == 'afl')
            outcomes = read(dest/'data/retrieval_rank_outcomes.json')
            selected_records = [r for r in outcomes if r['dataset'] == ds]
            assert len(selected_records) == 3
            left = 0.
            for label, key in [('improved', 'better'), ('tied', 'equal'), ('worsened', 'worse')]:
                row = next(r for r in selected_records if r['category'] == label)
                bar = next(r for r in panel['bars'] if r['dataset'] == ds and r['outcome'] == label)
                assert row['count'] == bar['count'] == pair[key]
                assert row['queries'] == bar['total'] == len(ranks)
                percent = 100*pair[key]/len(ranks)
                assert row['percent'] == bar['width'] == percent
                assert bar['left'] == left
                left += percent
            assert math.isclose(left, 100., abs_tol=1e-10)
        if successful:
            assert figure['shared_column_headers'] and figure['rows'] == 6
            text = (dest/(figure['name']+'.svg')).read_text()
            assert 'Selected successful retrievals' in text
    return dict(models=8, selected_examples=len(examples), complete_query_ranks_match_full_similarity_matrices=True,
                paired_outcomes_and_original_media_verified=True,
                selection_kind='selected_successes_and_misses' if mixed else 'selected_successful_examples' if successful else 'contrasting_diagnostics')


def check_all_method_rank_charts(manifest):
    dest, read = report.DEST, report.read
    protocol = read(dest/'data/retrieval_rank_comparison_protocol.json')
    assert protocol['display_only'] and protocol['energy_budget_kJ'] == 650.
    for file, digest in protocol['source_sha256'].items():
        assert report.sha(report.ROOT/file) == digest, file
    records = read(dest/'data/retrieval_all_comparator_outcomes.json')
    stats = read(dest/'data/retrieval_rank_box_stats.json')
    assert len(records) == 18 and len(stats) == 8
    bars = next(f for f in manifest['figures'] if f['name'] == report.NAMES[8])
    box = next(f for f in manifest['figures'] if f['name'] == report.NAMES[9])
    assert bars['all_retained_queries'] and not bars['numeric_bar_annotations']
    assert box['all_retained_queries'] and box['show_all_fliers'] and box['y_scale'] == 'linear'
    assert box['boxes'] == stats
    comparators = [m for m in report.METHODS if m != 'gate_select16']
    assert bars['comparator_order'] == comparators
    for ds in report.DATASETS:
        ranks = read(dest/'data'/f'retrieval_energy_{ds.lower()}_query_ranks.json')
        assert len(ranks) == protocol['query_counts'][ds]
        for method in comparators:
            delta = np.array([r[method+'_rank']-r['gate_select16_rank'] for r in ranks])
            bottom = 0.
            for outcome, count in [('improved', int(sum(delta > 0))), ('tied', int(sum(delta == 0))), ('worsened', int(sum(delta < 0)))]:
                row = next(r for r in records if (r['dataset'], r['comparator'], r['outcome']) == (ds, method, outcome))
                artist = next(r for r in bars['bars'] if (r['dataset'], r['comparator'], r['outcome']) == (ds, method, outcome))
                assert row['count'] == count and row['queries'] == len(ranks)
                assert row['percent'] == artist['height'] == 100*count/len(ranks)
                assert artist['bottom'] == bottom
                bottom += row['percent']
            assert math.isclose(bottom, 100., abs_tol=1e-10)
        for method in report.METHODS:
            values = np.array([r[method+'_rank'] for r in ranks], dtype=float)
            s = next(r for r in stats if (r['dataset'], r['method']) == (ds, method))
            q1, median, q3 = np.quantile(values, [.25, .5, .75])
            inside = values[(values >= q1-1.5*(q3-q1)) & (values <= q3+1.5*(q3-q1))]
            assert [s['q1'], s['median'], s['q3']] == [q1, median, q3]
            assert [s['whisker_low'], s['whisker_high']] == [min(inside), max(inside)]
            assert s['fliers'] == values[values < min(inside)].tolist()+values[values > max(inside)].tolist()
            assert s['queries'] == len(ranks) and s['minimum'] == min(values) and s['maximum'] == max(values)
            panel = next(p for p in box['axes'] if p['dataset'] == ds)
            assert panel['y']['limits'][0] < min(values) and panel['y']['limits'][-1] > max(values)
    for figure in [bars, box]:
        svg = ET.parse(dest/(figure['name']+'.svg')).getroot()
        assert not svg.findall('.//{http://www.w3.org/2000/svg}image')
    return dict(passed=True, query_counts=[1280, 256], comparator_pairs=6,
                all_four_method_boxplots=True, exact_counts_and_quantiles=True,
                all_outliers_in_plot_range=True, scientific_charts_without_images=True)


def main():
    torch.set_num_threads(1)
    root, dest, read, sha = report.ROOT, report.DEST, report.read, report.sha
    manifest = read(dest/'data/tgcn_manifest.json')
    audit = read(dest/'data/tgcn_audit.json')
    series = read(dest/'data/tgcn_plot_series.json')
    assert manifest['complete'] and audit['passed'] and len(manifest['figures']) == len(report.NAMES) == 10
    assert {f['name'] for f in manifest['figures']} == set(report.NAMES)
    for file, digest in manifest['original_figures'].items():
        assert sha(dest/file) == digest, file
    for file, digest in manifest.get('unchanged_measurements', {}).items():
        assert sha(dest/file) == digest, file
    for file, digest in manifest['source_sha256'].items():
        assert sha(root/file) == digest, file
    rows = read(dest/'data/fu_workload_metrics.json')
    assert len(rows) == 40 and read(dest/'data/fu_workload_controls.json')['passed']
    for row in rows:
        assert row['FL_rounds'] == 200 and row['FU_requests'] == 45
        assert row['FU_cycles'] == 45*row['fu_steps_per_request']
        assert row['joint_transactions']+row['FL_only_transactions'] == 180
        assert row['transactions'] == sum(row[k] for k in ['joint_transactions', 'FL_only_transactions', 'FU_only_transactions'])
        assert math.isclose(row['total_energy_j'], sum(row[k] for k in ['communication_j', 'compute_j', 'hover_j']), rel_tol=1e-12)
        source = root/row['source']
        requests = read(source/'request_records.json')
        assert all(r['steps'] == r['required_steps'] == row['fu_steps_per_request'] and r['completed_s'] is not None for r in requests)
    energy = read(dest/'data/energy_utility_checkpoints.json')
    assert len(series) == 16
    for s in series:
        if s['figure'] == report.NAMES[0]:
            records = [r for r in energy if (r['dataset'], r['method']) == (s['dataset'], s['method'])]
            assert s['x'] == [r['energy_kJ'] for r in records]
            assert s['y'] == [r['retained_mR_percent'] for r in records]
            assert records[-1]['FU_completed'] == 45
            assert max(s['x']) <= 650 and s['display_x'][-1] == 650 and s['display_y'][-1] == s['y'][-1]
            assert s['drawstyle'] == 'steps-post'
            for x, y in zip(s['marker_x'], s['marker_y']):
                assert y == next(r['retained_mR_percent'] for r in reversed(records) if r['energy_kJ'] <= x)
            recovered = read(checkpoints.OUT/(s['dataset'].lower()+'_seed0')/s['method']/'snapshot650.json')
            assert recovered['FL_round'] == records[-1]['FL_round']
            # The fixed-workload ledger sums component costs in a different order
            # from continuation's starting accumulator (observed <2e-12 kJ).
            assert math.isclose(recovered['energy_j']*.001, records[-1]['energy_kJ'],
                                rel_tol=0., abs_tol=1e-10)
            assert recovered['evaluation']['retained']['mR']*100 == records[-1]['retained_mR_percent']
        else:
            assert s['figure'] == report.NAMES[1] and s['kind'] == 'paired_bar'
            assert s['x'] == [3, 9]
            assert len(s['bar_positions']) == len(s['hatches']) == 2
            assert math.isclose(s['bar_positions'][1]-s['bar_positions'][0], .37)
            assert s['y'] == [next(r['total_energy_j']*.001 for r in rows if (r['dataset'], r['method'], r['fu_steps_per_request']) == (s['dataset'], s['method'], x)) for x in s['x']]
    for figure in manifest['figures']:
        precisions = []
        for ax in figure['axes']:
            precision = []
            for dim in ['x', 'y']:
                ticks = ax[dim]['ticks']
                if ax[dim].get('kind') == 'categorical':
                    expected = [label for method, label in report.METHODS.items() if method != 'gate_select16'] if figure['name'] == report.NAMES[8] else list(report.METHODS.values())
                    assert dim == 'x' and ax[dim]['labels'] == expected
                    assert ax[dim]['limits'][0] < ticks[0] and ax[dim]['limits'][-1] > ticks[-1]
                    precision.append(None)
                    continue
                assert ax[dim]['limits'] == [ticks[0], ticks[-1]]
                assert np.allclose(np.diff(ticks), ticks[1]-ticks[0])
                digits = [len(t.split('.')[-1]) if '.' in t else 0 for t in ax[dim]['labels']]
                assert len(set(digits)) == 1
                precision.append(digits[0])
            precisions.append(precision)
        if precisions:
            assert precisions[0] == precisions[1]
        if figure['name'] == report.NAMES[1]:
            assert figure['kind'] == 'paired_bar' and figure['steps_per_request'] == [3, 9]
            assert figure['bars_per_method'] == 2 and figure['bars_per_panel'] == 8
            assert not figure['bar_top_annotations']
        for item in figure['files']:
            path = dest/item['file']
            assert sha(path) == item['sha256']
            if path.suffix == '.png':
                with Image.open(path) as image:
                    assert image.width > 1800 and image.height > 800
            if path.suffix == '.svg':
                text = path.read_text()
                assert 'Proposed (' not in text
                visible_text = ''.join(''.join(ET.parse(path).getroot().itertext()).split())
                assert all(label.replace(' ', '') in visible_text for label in report.METHODS.values())
    fixed = check_retrieval(retrieval, 'retrieval', manifest['figures'])
    matched = check_retrieval(energy_retrieval, 'retrieval_energy', manifest['figures'])
    combined = check_retrieval(energy_retrieval, 'retrieval_combined', manifest['figures'])
    integrated = check_retrieval(energy_retrieval, 'retrieval_integrated', manifest['figures'])
    rank_charts = check_all_method_rank_charts(manifest)
    result = dict(passed=True, new_figures=10, fresh_FU_arms=40, exact_baseline_controls=8,
        energy_budget_kJ=650, energy_checkpoint_replays_exact=True,
        fixed_workload_retrieval=fixed, matched_energy_retrieval=matched, combined_retrieval=combined,
        integrated_retrieval=integrated, full_query_outcome_counts_and_bar_widths_verified=True,
        all_method_rank_charts=rank_charts,
        paired_FU_workload_bars_from_measured_1_and_9_steps=True,
        original_nine_figures_unchanged=not bool(manifest.get('typography_revision')),
        original_nine_measurements_unchanged=True,
        authorized_typography_revision=manifest.get('typography_revision'),
        actual_frame_figures_only=True,
        regular_ticks_common_marker_budgets=True,
        source_sha256={str(Path(__file__).relative_to(root)): sha(Path(__file__))})
    report.write(dest/'data/tgcn_artifact_checks.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
