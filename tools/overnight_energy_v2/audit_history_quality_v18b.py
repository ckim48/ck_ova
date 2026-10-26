"""Check all saved pilot vectors/probabilities and expose both KL references."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
out = root/'results/overnight_energy_v2/history_quality_v18b'
assert json.loads((out/'status.json').read_text())['variants'] == 7
plan = json.loads((out/'plan.json').read_text())
rows = json.loads((out/'results.json').read_text())
groups = json.loads((root/'results/overnight_energy_v2/single_step_quality_pilot/cohorts.json').read_text())
quality = root/'results/energy_quality_v1'
full_groups = json.loads((quality/'cohorts.json').read_text())['groups']
reference = {}; labels = {}
with np.load(quality/'retrain_reference_ideal.npz') as data:
    for name, spec in groups.items():
        lookup = {value: k for k, value in enumerate(full_groups[name]['ids'])}
        indices = [lookup[z] for z in spec['ids']]
        reference[name] = data[name][indices].copy()
        labels[name] = data[name+'_labels'][indices].copy()
with np.load(out/'ideal_one_step.npz') as data:
    ideal = {name: data[name].copy() for name in groups}
certificates = []
with np.load(out/'update_vectors.npz') as vectors:
    for row in rows:
        name = row['method']
        assert hashlib.sha256(np.ascontiguousarray(vectors[name]).tobytes()).hexdigest() == plan['variants'][name]['update_sha256']
        with np.load(out/(name+'.npz')) as data:
            for group in groups:
                p = data[group].copy(); q = reference[group]; i = ideal[group]
                assert p.shape == q.shape == (200, 100)
                np.testing.assert_allclose(p.sum(axis=1), 1., rtol=1e-12, atol=1e-14)
                assert np.all(np.isfinite(p)) and np.all(p > 0)
                loss = float(-np.log(p[np.arange(200), labels[group]]).mean())
                accuracy = float(np.mean(p.argmax(axis=1) == labels[group]))
                kli = float(np.mean(np.sum(i*(np.log(i)-np.log(p)), axis=1)))
                klr = float(np.mean(np.sum(q*(np.log(q)-np.log(p)), axis=1)))
                np.testing.assert_allclose([loss, accuracy, max(kli, 0.), max(klr, 0.)],
                    [row[group]['loss'], row[group]['accuracy'], row[group]['gap_to_ideal_step']['kl_reference_to_candidate'],
                     row[group]['gap_to_final_retrain_descriptive_only']['kl_reference_to_candidate']], rtol=1e-12, atol=1e-14)
                certificates.append(dict(method=name, cohort=group, loss=loss, accuracy=accuracy,
                    ideal_step_kl=max(kli, 0.), final_retrain_kl_descriptive_only=max(klr, 0.)))
summary = []
for family in ['unprojected_baseline', 'recent_public_span_rank3']:
    for group in groups:
        eligible = [x for x in certificates if x['method'].startswith(family+'_') and x['cohort'] == group]
        summary.append(dict(family=family, cohort=group, noise_draws=len(eligible),
            **{name+'_mean': float(np.mean([x[name] for x in eligible])) for name in ['loss', 'accuracy', 'ideal_step_kl', 'final_retrain_kl_descriptive_only']}))
result = dict(status='passed_all7_vectors_and14_cohort_predictions', at_kst=datetime.now().isoformat(timespec='seconds'),
    certificates=certificates, summary=summary,
    interpretation='All variants preselected. Rank3 has slightly lower mean loss and ideal-step KL on this one learning-only fixture, but slightly larger final-retraining KL. Round30 precedes FU; neither direction of its final-reference KL is evidence of unlearning. All raw values retained. No test arrays, new model inference or inference of full-trajectory equivalence.',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'independent_metrics.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k != 'certificates'}))
