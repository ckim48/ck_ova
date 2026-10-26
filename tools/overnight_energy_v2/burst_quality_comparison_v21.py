"""Non-test cache-only comparison of completed original and changed Hybrid."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
canonical = root/'results/energy_quality_v1'; burst = root/'results/overnight_energy_v2/burst_joint_v3'
out = root/'results/overnight_energy_v2/burst_quality_comparison_v21'; out.mkdir(exist_ok=True)
assert not (out/'results.json').exists(), 'Do not duplicate completed cache analysis.'
old = json.loads((canonical/'coupled_hybrid.cache.json').read_text())
new = json.loads((burst/'quality/hybrid.cache.json').read_text())
ref = json.loads((canonical/'retrain_reference_ideal.cache.json').read_text())
assert old['cohorts_sha256'] == new['cohorts_sha256'] == ref['cohorts_sha256'] == sha(canonical/'cohorts.json')
assert old['initial_checkpoint_sha256'] == new['initial_checkpoint_sha256']
assert old['evaluator_sha256'] == new['numeric_evaluator_sha256'] == ref['evaluator_sha256']
assert old['checkpoint_sha256'] == sha(root/'results/comparisons_seed0_solver_v2/coupled_hybrid/latest.pt')
assert new['checkpoint_sha256'] == sha(burst/'hybrid/latest.pt')
old_row = next(x for x in json.loads((canonical/'energy_quality_report.json').read_text())['methods'] if x['method'] == 'coupled_hybrid')
new_row = next(x for x in json.loads((burst/'quality/report.json').read_text())['methods'] if x['method'] == 'burst_joint_v3/hybrid')
assert old_row['status'] == new_row['status'] == 'completed' and new_row['quality_status'] == 'evaluated'
paths = [canonical/'coupled_hybrid.npz', burst/'quality/hybrid.npz', canonical/'retrain_reference_ideal.npz']
groups = ['validation', 'retained', 'forget', 'nonmember_validation']; caches = []
for path in paths:
    with np.load(path) as data:
        caches.append({name: data[name].copy() for group in groups for name in [group, group+'_labels']})
repetitions = 5000; rng = np.random.default_rng(2026102109); rows = []; differences = {}
for group in ['validation', 'retained', 'forget']:
    labels = caches[0][group+'_labels']
    assert all(np.array_equal(labels, cache[group+'_labels']) for cache in caches)
    pold, pnew, pref = [np.maximum(cache[group], 1e-300) for cache in caches]
    a = -np.log(pold[np.arange(len(labels)), labels]); b = -np.log(pnew[np.arange(len(labels)), labels])
    ka = np.sum(pref*(np.log(pref)-np.log(pold)), axis=1)
    kb = np.sum(pref*(np.log(pref)-np.log(pnew)), axis=1)
    np.testing.assert_allclose([a.mean(), b.mean(), ka.mean(), kb.mean()],
        [old_row[group+'_loss'], new_row[group+'_loss'], old_row[group+'_kl_reference_to_candidate'], new_row[group+'_kl_reference_to_candidate']], rtol=1e-12, atol=1e-14)
    classes = [np.flatnonzero(labels == z) for z in np.unique(labels)]
    indices = np.concatenate([rng.choice(z, (repetitions, len(z)), replace=True) for z in classes], axis=1)
    for metric, delta in [('true_label_loss', b-a), ('kl_to_final_retrain', kb-ka),
        ('reference_prediction_disagreement', (pnew.argmax(axis=1) != pref.argmax(axis=1)).astype(float)-(pold.argmax(axis=1) != pref.argmax(axis=1)))]:
        means = delta[indices].mean(axis=1)
        rows.append(dict(cohort=group, metric=metric, changed_minus_original_mean=float(delta.mean()),
                         conditional_percentile_interval_95=np.quantile(means, [.025, .975]).tolist()))
        differences[group+'_'+metric] = delta
old_rounds = [json.loads(x) for x in (root/'results/comparisons_seed0_solver_v2/coupled_hybrid/rounds.jsonl').read_text().splitlines() if x.strip()]
new_rounds = [json.loads(x) for x in (burst/'hybrid/rounds.jsonl').read_text().splitlines() if x.strip()]
assert len(old_rounds) == len(new_rounds) == 100
assert all(x['round'] == y['round'] and x['newly_deleted'] == y['newly_deleted'] for x, y in zip(old_rounds, new_rounds))
plan = dict(code_sha256=sha(__file__), old_cache_metadata=old, new_cache_metadata=new, reference_metadata=ref,
            prediction_cache_sha256={str(p.relative_to(root)): sha(p) for p in paths}, bootstrap_repetitions=repetitions,
            bootstrap_seed=2026102109, comparison='Changed protocol minus original Hybrid; conditional sample analysis, not matched-constraint superiority or a causal budget-only ablation.',
            limits=['Changed run uses request RMS0.6 and joint-power solver, original uses0.2/frozen-power solver. Trajectories differ, so savings cannot be attributed solely to an epsilon or solver change.',
                    'Fixed final models, one training seed, paired class-stratified1000-sample cohorts. Intervals describe image-sample uncertainty, not seed/client/radio uncertainty or confirmatory testing.',
                    'No test arrays or new inference used; no settings chosen from these intervals. Both complete100 rounds/5request deliveries; delivery is not deletion certification.',
                    'Modeled communication upper bound, not measured total UAV energy. Same changed-envelope Mixed/optimized-phase comparisons remain pending.'])
(out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
result = dict(status='completed_cache_only_changed_protocol_comparison', at_kst=datetime.now().isoformat(timespec='seconds'),
    original=old_row, changed=new_row, energy_upper_reduction_fraction=1-new_row['energy_bound_j']/old_row['energy_bound_j'],
    original_request_round_energy_j=sum(x['transmission']['energy_bound_j'] for x in old_rounds if x['newly_deleted'] > 0),
    changed_request_round_energy_j=sum(x['transmission']['energy_bound_j'] for x in new_rounds if x['newly_deleted'] > 0),
    rows=rows, limits=plan['limits'])
np.savez_compressed(out/'per_sample_differences.npz', **differences)
(out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ['original', 'changed']}))
