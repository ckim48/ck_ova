"""Check scheduling evidence, matched costs and unsmoothed progress figures."""
from pathlib import Path
import math
import numpy as np
from PIL import Image
import parallel_advantage_report as report


def main():
    read, sha, dest = report.read, report.sha, report.DEST
    report.study.register()
    manifest = read(dest / 'data/manifest.json')
    rows = read(dest / 'data/metrics.json')
    series = read(dest / 'data/plot_series.json')
    budget = read(dest / 'data/budget400.json')
    audit = read(dest / 'data/audit.json')
    assert manifest['complete'] and len(manifest['figures']) == 5 and len(rows) == 4
    assert audit['passed'] and audit['matched_control']['original_PKS_reproduced_exactly']
    assert len(audit['continuation_checks']) == len(budget) == 8
    by = {(r['dataset'], r['method']): r for r in rows}
    original = read(report.ROOT / 'final_final_charts/proposed_pks/data/metrics.json')
    previous_progress = read(report.ROOT / 'final_final_charts/same_time/data/metrics.json')
    for r in rows:
        assert r['FL_rounds'] == 200 and r['total_FU_steps'] == r['FU_cycles'] == 225
        assert r['queue_wait_s'] == 0
        if r['method'] == 'proposed':
            assert (r['fl'], r['joint'], r['fu'], r['broadcasts'], r['joint_FU_steps']) == (135, 45, 0, 179, 225)
            old = next(v for v in original if v['case'] == 'baseline' and v['dataset'] == r['dataset'] and v['method'] == 'gate_select16')
            for metric in ['delay_s', 'total_energy_j', 'retained_mR_percent', 'retained_loss', 'forget_KL', 'worst_request_KL']:
                assert r[metric] == old[metric], (r['dataset'], metric)
        else:
            assert (r['fl'], r['joint'], r['fu'], r['broadcasts'], r['joint_FU_steps']) == (180, 0, 45, 224, 0)
    for b in budget:
        old_id = 'gate_select16' if b['method'] == 'proposed' else b['method']
        old = next(r for r in previous_progress if r['dataset'] == b['dataset'] and r['method'] == old_id and r['budget_s'] == 400)
        assert b['active_s'] <= 400 and b['FL_round'] == old['completed_FL_round']
        assert b['FU_cycles'] == old['completed_FU_cycles'] == 225
    for s in series:
        if s.get('kind') == 'service_window':
            assert s['starting_FL_round'] == 25 and s['extra_FL_rounds'] == 5 and s['FU_steps'] == 25
            events = s['events']
            assert sum(e['FL'] for e in events) == sum(e['FU'] for e in events) == 5
            assert math.isclose(events[0]['start_s'], 0, abs_tol=1e-9)
            assert events[-1]['end_s'] == s['duration_s']
            for a, b in zip(events, events[1:]):
                assert math.isclose(a['end_s'], b['start_s'], abs_tol=1e-9)
            if s['method'] == 'proposed':
                assert len(events) == 5 and all(e['FL'] and e['FU'] for e in events)
            else:
                assert len(events) == 10 and all(e['FL'] != e['FU'] for e in events)
            assert s['duration_s'] == by[(s['dataset'], s['method'])]['first_release_plus_five_FL_s']
        elif s.get('bar'):
            assert s['y'] == [by[(s['dataset'], s['method'])][s['metric']] * s['scale']]
        else:
            assert s['drawstyle'] == 'steps-post' and len(s['x']) == len(s['y'])
            assert all(a <= b for a, b in zip(s['x'], s['x'][1:]))
            assert all(a <= b for a, b in zip(s['y'], s['y'][1:]))
            assert s['display_x'][-1] == 400 and s['display_y'][-1] == s['y'][-1]
            b = next(b for b in budget if b['dataset'] == s['dataset'] and b['method'] == s['method'])
            assert s['y'][-1] == b[s['metric']]
    for f in manifest['figures']:
        precisions = []
        for panel in f['axes']:
            result = {}
            for dimension in ['x', 'y']:
                a = panel[dimension]
                if a.get('kind') == 'categorical':
                    assert a['labels'] == ['S-PKS', 'Proposed']
                    continue
                ticks = a['ticks']
                assert a['limits'] == [ticks[0], ticks[-1]] and np.allclose(np.diff(ticks), ticks[1] - ticks[0])
                digits = [len(v.split('.')[-1]) if '.' in v else 0 for v in a['labels']]
                assert len(set(digits)) == 1
                result[dimension] = digits[0]
            precisions.append(result)
        assert precisions[0] == precisions[1]
        for item in f['files']:
            path = dest / item['file']
            assert sha(path) == item['sha256']
            if path.suffix == '.png':
                with Image.open(path) as image:
                    assert image.width > 1800 and image.height > 800
    for name, digest in manifest['source_sha256'].items():
        assert sha(report.ROOT / name) == digest, name
    result = dict(passed=True, figures=5, fresh_ablation_arms=4,
                  identical_PKS_control_verified=True, full_workload_and_window_counts_verified=True,
                  immediate_serial_without_waiting=True, cost_and_quality_values_preserved=True,
                  actual_400_second_progress_preserved=True, plot_values_ticks_and_hashes_verified=True,
                  source_sha256={str(Path(__file__).relative_to(report.ROOT)): sha(Path(__file__))})
    report.write(dest / 'data/artifact_checks.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
