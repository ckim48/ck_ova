"""Paired final-model diagnostic using frozen non-test prediction caches only."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def indices(labels, repetitions, rng):
    groups = [np.flatnonzero(labels == x) for x in np.unique(labels)]
    return np.concatenate([rng.choice(ids, (repetitions, len(ids)), replace=True) for ids in groups], axis=1)


def auc(a, b):
    b = np.sort(b)
    return float(np.mean((np.searchsorted(b, a, 'left')+np.searchsorted(b, a, 'right'))/(2*len(b))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    root = p.parse_args().project.resolve()
    source = root/'results/energy_quality_v1'
    out = root/'results/overnight_energy_v2/retain_control_audit_v16'; out.mkdir(exist_ok=True)
    assert not (out/'results.json').exists(), 'Reuse completed audit; no duplicate inference or analysis.'
    names = ['coupled_ideal', 'retain_only_ideal', 'retrain_reference_ideal']
    groups = ['validation', 'retained', 'forget', 'nonmember_validation']
    metadata = {n: json.loads((source/(n+'.cache.json')).read_text()) for n in names}
    cohort_sha = sha(source/'cohorts.json')
    for n in names:
        m = metadata[n]
        assert m['cohorts_sha256'] == cohort_sha
        assert m['evaluator_sha256'] == metadata[names[0]]['evaluator_sha256']
        assert m['initial_checkpoint_sha256'] == metadata[names[0]]['initial_checkpoint_sha256']
        assert m['training_source_sha256'] == metadata[names[0]]['training_source_sha256']
        assert m['checkpoint_sha256'] == sha(root/'results/comparisons_seed0_solver_v2'/n/'latest.pt')
    caches = {}
    for n in names:
        with np.load(source/(n+'.npz')) as data:
            # No test or nonmember-test probabilities enter this analysis.
            caches[n] = {k: data[k].copy() for group in groups for k in [group, group+'_labels']}
    canonical = {x['method']: x for x in json.loads((source/'energy_quality_report.json').read_text())['methods']}
    seed = 20261021; repetitions = 5000; mia_repetitions = 2000
    plan = dict(script_sha256=sha(__file__), cache_metadata=metadata, cohorts_sha256=cohort_sha,
                prediction_cache_sha256={n: sha(source/(n+'.npz')) for n in names}, seed=seed,
                paired_stratified_bootstrap_repetitions=repetitions, mia_bootstrap_repetitions=mia_repetitions,
                comparison='coupled_ideal MINUS retain_only_ideal; lower loss/KL/disagreement/gap is better',
                limits=['Fixed final models, one training seed, frozen1000-sample cohorts, class-stratified paired resampling. Descriptive conditional sample uncertainty only, not training-seed or radio uncertainty, not a confirmatory test.',
                        'Within-class image resampling does not model client dependence. No hyperparameters/configurations selected from these intervals.',
                        'Both ideal-channel controls are reference-only; zero modeled energy is not a physical competitor. Excluding deleted data from future SGD is not certified deletion.',
                        'MIA score is negative true-label loss; chance AUC or small differences do not prove unlearning. No test arrays or new GPU inference used.'])
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    rng = np.random.default_rng(seed); rows = []; saved = {}; losses = {}
    def summarize(name, group, delta, boot_indices):
        means = delta[boot_indices].mean(axis=1)
        rows.append(dict(metric=name, cohort=group, samples=len(delta), coupled_minus_retain_only=float(delta.mean()),
                         conditional_percentile_interval_95=np.quantile(means, [.025, .975]).tolist()))
        saved[group+'_'+name+'_delta'] = delta
    for group in groups:
        labels = caches[names[0]][group+'_labels']
        assert all(np.array_equal(labels, caches[n][group+'_labels']) for n in names)
        for n in names:
            prob = np.maximum(caches[n][group], 1e-300)
            losses[n, group] = -np.log(prob[np.arange(len(labels)), labels])
        if group == 'nonmember_validation':
            continue
        idx = indices(labels, repetitions, rng)
        a, b, ref = [np.maximum(caches[n][group], 1e-300) for n in names]
        kl_a = np.sum(ref*(np.log(ref)-np.log(a)), axis=1)
        kl_b = np.sum(ref*(np.log(ref)-np.log(b)), axis=1)
        np.testing.assert_allclose(kl_a.mean(), canonical[names[0]][group+'_kl_reference_to_candidate'], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(kl_b.mean(), canonical[names[1]][group+'_kl_reference_to_candidate'], rtol=1e-12, atol=1e-14)
        summarize('kl_to_final_retrain', group, kl_a-kl_b, idx)
        summarize('true_label_loss', group, losses[names[0], group]-losses[names[1], group], idx)
        summarize('reference_prediction_disagreement', group,
                  (a.argmax(axis=1) != ref.argmax(axis=1)).astype(float)-(b.argmax(axis=1) != ref.argmax(axis=1)), idx)
    fi = indices(caches[names[0]]['forget_labels'], mia_repetitions, rng)
    ni = indices(caches[names[0]]['nonmember_validation_labels'], mia_repetitions, rng)
    point = {n: auc(-losses[n, 'forget'], -losses[n, 'nonmember_validation']) for n in names}
    for n in names:
        np.testing.assert_allclose(point[n], canonical[n]['forget_loss_attack_auc_validation'], rtol=1e-12, atol=1e-14)
    auc_delta = []; gap_delta = []
    for f, v in zip(fi, ni):
        values = [auc(-losses[n, 'forget'][f], -losses[n, 'nonmember_validation'][v]) for n in names]
        auc_delta.append(values[0]-values[1])
        gap_delta.append(abs(values[0]-values[2])-abs(values[1]-values[2]))
    result = dict(status='completed_cache_only_non_test_paired_audit', at_kst=datetime.now().isoformat(timespec='seconds'),
                  rows=rows, mia=dict(point_auc=point, coupled_minus_retain_auc=point[names[0]]-point[names[1]],
                    auc_delta_conditional_interval_95=np.quantile(auc_delta, [.025, .975]).tolist(),
                    coupled_minus_retain_absolute_auc_gap=abs(point[names[0]]-point[names[2]])-abs(point[names[1]]-point[names[2]]),
                    absolute_gap_delta_conditional_interval_95=np.quantile(gap_delta, [.025, .975]).tolist()),
                  plan_sha256=sha(out/'plan.json'), interpretation=plan['limits'])
    np.savez_compressed(out/'per_sample_differences.npz', **saved,
                        auc_delta_bootstrap=np.array(auc_delta), absolute_auc_gap_delta_bootstrap=np.array(gap_delta))
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
