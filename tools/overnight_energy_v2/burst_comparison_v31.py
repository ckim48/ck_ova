"""Completed matched burst Hybrid/Mixed non-test cache comparison."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK');f=root/'results/overnight_energy_v2'
burst=f/'burst_joint_v3';quality=burst/'quality';canonical=root/'results/energy_quality_v1';out=f/'burst_comparison_v31';out.mkdir(exist_ok=True)
assert not (out/'results.json').exists(),'No duplicate cache comparison.'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=json.loads((quality/'report.json').read_text());observed={x['method']:x for x in report['methods']}
names=['hybrid','mixed_aircomp'];selected=[observed['burst_joint_v3/'+name] for name in names]
assert all(x['status']=='completed' and x['quality_status']=='evaluated' and x['round']==100 and x['deletion_rounds_delivered']==5 for x in selected)
meta=[json.loads((quality/(name+'.cache.json')).read_text()) for name in names]
assert meta[0]['cohorts_sha256']==meta[1]['cohorts_sha256']==sha(canonical/'cohorts.json')
assert meta[0]['initial_checkpoint_sha256']==meta[1]['initial_checkpoint_sha256']
assert meta[0]['numeric_evaluator_sha256']==meta[1]['numeric_evaluator_sha256']
assert all(m['checkpoint_sha256']==sha(burst/name/'latest.pt') for name,m in zip(names,meta))
trajectories=[]
for name in names:
    rows=[json.loads(x) for x in (burst/name/'rounds.jsonl').read_text().splitlines() if x.strip()];assert len(rows)==100
    trajectories.append(rows)
assert [(x['round'],x['newly_deleted']) for x in trajectories[0]]==[(x['round'],x['newly_deleted']) for x in trajectories[1]]
paths=[quality/(name+'.npz') for name in names]+[canonical/'retrain_reference_ideal.npz'];caches=[]
for path in paths:
    with np.load(path) as data:caches.append({k:data[k].copy() for g in ['validation','retained','forget'] for k in [g,g+'_labels']})
plan=dict(code_sha256=sha(__file__),at_kst=datetime.now().isoformat(timespec='seconds'),comparison='Mixed minus Hybrid in same changed burst protocol',
    metadata=meta,prediction_sha256={str(p.relative_to(root)):sha(p) for p in paths},bootstrap_repetitions=5000,bootstrap_seed=202610211131,
    limits=['Same changedrequest0.6/normal0.2 policy and jointpower solver, initialmodel/schedule/cohort/numeric evaluator; both100rounds and5request deliveries. Resulting model/update trajectories differ, as expected in fulltraining.',
        'Paired class-stratified image resampling conditional on two fixedmodels from one trainingseed. Not seed/client/radio uncertainty, confirmatory significance or quality equivalence.',
        'No new inference/test arrays/settingsselection. Accuracy/prediction-disagreement/MIA gaps retained alongside loss/energy. Delivery and smallMSE/MIA are not deletion certification.',
        'Modeled communication upper bound, not measuredUAVtotal energy. Optimizedphase fulltrajectory is nowrunning and excluded from completed-energy ranking.'])
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
rng=np.random.default_rng(plan['bootstrap_seed']);results=[];differences={}
for group in ['validation','retained','forget']:
    labels=caches[0][group+'_labels'];assert all(np.array_equal(labels,x[group+'_labels']) for x in caches)
    ph,pm,pref=[x[group] for x in caches];n=len(labels)
    kl=lambda p:np.sum(pref*(np.log(pref)-np.log(p)),axis=1)
    metrics=dict(true_label_loss=-np.log(pm[np.arange(n),labels])+np.log(ph[np.arange(n),labels]),
        kl_to_final_retrain=kl(pm)-kl(ph),accuracy=(pm.argmax(axis=1)==labels).astype(float)-(ph.argmax(axis=1)==labels),
        reference_prediction_disagreement=(pm.argmax(axis=1)!=pref.argmax(axis=1)).astype(float)-(ph.argmax(axis=1)!=pref.argmax(axis=1)))
    classes=[np.flatnonzero(labels==z) for z in np.unique(labels)];indices=np.concatenate([rng.choice(z,(5000,len(z)),replace=True) for z in classes],axis=1)
    for metric,delta in metrics.items():
        means=delta[indices].mean(axis=1);results.append(dict(cohort=group,metric=metric,mixed_minus_hybrid=float(delta.mean()),conditional_percentile_interval95=np.quantile(means,[.025,.975]).tolist()))
        differences[group+'_'+metric]=delta
old=next(x for x in json.loads((canonical/'energy_quality_report.json').read_text())['methods'] if x['method']=='coupled_mixed_aircomp')
result=dict(status='completed_matched_burst_non_test_cache_comparison',at_kst=datetime.now().isoformat(timespec='seconds'),
    methods=[{k:v for k,v in row.items() if not k.startswith('test') and not k.endswith('_test')} for row in selected],
    mixed_energy_reduction_vs_same_protocol_hybrid=1-selected[1]['energy_bound_j']/selected[0]['energy_bound_j'],paired=results,
    changed_mixed_vs_original_mixed=dict(energy_upper_reduction_fraction=1-selected[1]['energy_bound_j']/old['energy_bound_j'],
        validation_loss_change=selected[1]['validation_loss']-old['validation_loss'],retained_loss_change=selected[1]['retained_loss']-old['retained_loss'],
        forget_kl_change=selected[1]['forget_kl_reference_to_candidate']-old['forget_kl_reference_to_candidate'],
        interpretation='Differentrequesttolerance AND solver/trajectory: descriptive energy-quality tradeoff, not causal budget-only ablation or matched-quality superiority.'),limits=plan['limits'])
np.savez_compressed(out/'paired_differences.npz',**differences)
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['methods','limits']}))
