"""Independent non-test cache arithmetic and completed energy/request totals."""
from datetime import datetime
import hashlib,json
from pathlib import Path
import numpy as np

r=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK');b=r/'results/overnight_energy_v2/burst_joint_v3';out=r/'results/overnight_energy_v2/burst_comparison_v31'
assert not (out/'certificate.json').exists()
report=json.loads((b/'quality/report.json').read_text());records={x['method']:x for x in report['methods']};checked=[]
with np.load(r/'results/energy_quality_v1/retrain_reference_ideal.npz') as data:ref={k:data[k].copy() for g in ['validation','retained','forget'] for k in [g,g+'_labels']}
for method in ['hybrid','mixed_aircomp']:
    row=records['burst_joint_v3/'+method]
    with np.load(b/'quality'/(method+'.npz')) as data:
        for group in ['validation','retained','forget']:
            p=data[group];labels=data[group+'_labels'];np.testing.assert_array_equal(labels,ref[group+'_labels'])
            q=ref[group];n=len(labels);values=[float(-np.log(p[np.arange(n),labels]).mean()),float(np.mean(p.argmax(axis=1)==labels)),
                max(0.,float(np.mean(np.sum(q*(np.log(q)-np.log(p)),axis=1)))),float(np.mean(q.argmax(axis=1)!=p.argmax(axis=1)))]
            np.testing.assert_allclose(values,[row[group+'_loss'],row[group+'_accuracy'],row[group+'_kl_reference_to_candidate'],row[group+'_prediction_disagreement']],rtol=1e-12,atol=1e-14)
            checked.append(dict(method=method,cohort=group,loss=values[0],accuracy=values[1],reference_kl=values[2],reference_prediction_disagreement=values[3]))
    rounds=[json.loads(x) for x in (b/method/'rounds.jsonl').read_text().splitlines() if x.strip()]
    assert len(rounds)==100 and sum(x['newly_deleted']>0 for x in rounds)==5
    np.testing.assert_allclose(sum(x['transmission']['energy_bound_j'] for x in rounds),row['energy_bound_j'],rtol=1e-12,atol=1e-14)
certificate=dict(status='passed6_non_test_cohort_metric_sets_and_2_completed_energy_request_totals',at_kst=datetime.now().isoformat(timespec='seconds'),
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),report_sha256=hashlib.sha256((b/'quality/report.json').read_bytes()).hexdigest(),metrics=checked,
    limits='Cache arithmetic/provenance checks only; no new inference, test selection, fulltask quality equivalence, statistical seed evidence or deletion certification.')
(out/'certificate.json').write_text(json.dumps(certificate,indent=2)+'\n');print(json.dumps({k:v for k,v in certificate.items() if k!='metrics'}))
