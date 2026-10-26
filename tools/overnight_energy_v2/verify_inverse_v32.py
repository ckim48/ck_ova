"""Independent physical update reconstruction and cached output audit; no inference."""
from datetime import datetime
import hashlib,json
from pathlib import Path
import numpy as np

r=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK');f=r/'results/overnight_energy_v2';q=f/'inverse_quality_v30'
out=f/'inverse_quality_audit_v32';out.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
assert not (out/'certificate.json').exists()
assert read(q/'status.json')['status'].startswith('completed9_')
plan=read(q/'plan.json');cfg=read(f/'public_history_fu_v19/screen/plan.json')['config'];pilot=read(f/'fu_quality_v22/plan.json')
for n,h in plan['core_sha256'].items():assert sha(r/n)==h
assert sha(f/'inverse_heads_v29b/summary.json')==plan['complete61case_610head_summary_sha256']
with np.load(f/'public_history_fu_v19/screen/vectors.npz') as a:
    learning,unlearning,channels=[a[k].copy() for k in ['learning','unlearning','channels']];basis=a['rank3_basis'].copy();public_scale=float(a['public_scale'])
dimension=learning.shape[1];width=cfg['block_dimension'];sigma=np.sqrt(10**((cfg['noise_dbm']-30)/10))
block=lambda x:np.pad(x,((0,0),(0,(-dimension)%width))).reshape(len(x),-1,width)
vectors_checked=[]
with np.load(q/'update_vectors.npz') as vectors:
    assert len(vectors.files)==9
    for chosen in plan['selected']:
        path=r/chosen['source_design_path'];assert sha(path)==chosen['source_design_sha256']
        with np.load(path) as a:d={k:a[k].copy() for k in a.files}
        mode=chosen['mode'];raw=learning+unlearning if mode=='mixed_aircomp' else learning
        if chosen['rank']:
            coeff=((raw@basis)/public_scale).astype(np.float16).astype(float)*public_scale;raw=raw-coeff@basis.T
        analog=block(raw);digital=block(np.zeros_like(unlearning) if mode=='mixed_aircomp' else unlearning)
        levels=2**(chosen['bits']-1)-1;scale=(np.max(np.abs(digital),axis=-1)/levels).astype(np.float32).astype(float)
        decoded=np.clip(np.rint(digital/np.where(scale>0,scale,1.)[...,None]),-levels,levels)*scale[...,None]
        users=d['users'];rho=np.linalg.norm(analog[users],axis=-1);symbols=d['symbols']
        b=np.divide(d['analog_rms']*np.sqrt(symbols),rho,out=np.zeros_like(rho),where=rho>0)
        c=d['receiver']/(sigma*np.sqrt(symbols));h=channels[users,int(d['head']),None]*sigma
        bias=(((c*h*b-1)[...,None]*analog[users])+decoded[users]-digital[users]).sum(axis=0)
        for seed in plan['noise_seeds']:
            name=f'{mode}_noise{seed}';normal=np.random.default_rng(seed).normal(size=bias.shape)
            physical=(learning+unlearning).sum(axis=0)+(bias+c[:,None]*sigma/np.sqrt(2)*normal).reshape(-1)[:dimension]
            assert hashlib.sha256(np.ascontiguousarray(vectors[name]).tobytes()).hexdigest()==plan['update_sha256'][name]
            np.testing.assert_allclose(vectors[name],physical,rtol=1e-10,atol=1e-14);vectors_checked.append(name)
full=read(r/'results/energy_quality_v1/cohorts.json')['groups'];reference={};labels={};groups=plan['groups']
with np.load(r/'results/energy_quality_v1/retrain_reference_ideal.npz') as a:
    for g,s in groups.items():
        lookup={z:i for i,z in enumerate(full[g]['ids'])};indices=[lookup[z] for z in s['ids']]
        reference[g]=a[g][indices];labels[g]=a[g+'_labels'][indices]
with np.load(f/'fu_quality_v22/ideal_one_step.npz') as a:ideal={g:a[g].copy() for g in groups}
checks=[];rows=read(q/'results.json');assert len(rows)==9
for row in rows:
    path=q/(row['method']+'.npz')
    with np.load(path) as a:
        for g,s in groups.items():
            p=a[g];assert p.shape==(len(s['ids']),100) and np.all(np.isfinite(p)) and np.all(p>0)
            np.testing.assert_allclose(p.sum(axis=1),1.,rtol=1e-12,atol=1e-14)
            values=[float(-np.log(p[np.arange(len(p)),labels[g]]).mean()),float(np.mean(p.argmax(axis=1)==labels[g]))]
            np.testing.assert_allclose(values,[row[g]['loss'],row[g]['accuracy']],rtol=1e-12,atol=1e-14)
            for label,ref in [('ideal_step',ideal[g]),('final_retrain_descriptive_only',reference[g])]:
                kl=max(0.,float(np.mean(np.sum(ref*(np.log(ref)-np.log(p)),axis=1))));disagree=float(np.mean(ref.argmax(axis=1)!=p.argmax(axis=1)))
                np.testing.assert_allclose([kl,disagree],[row[g]['gap_to_'+label]['kl_reference_to_candidate'],row[g]['gap_to_'+label]['prediction_disagreement']],rtol=1e-12,atol=1e-14)
            checks.append(dict(method=row['method'],cohort=g,cache_sha256=sha(path)))
certificate=dict(status='passed_all9_physical_vectors_and27_non_test_cohort_metric_sets',at_kst=datetime.now().isoformat(timespec='seconds'),source_sha256=sha(Path(__file__)),plan_sha256=sha(q/'plan.json'),vectors=vectors_checked,cohorts=checks,limits='Cached arithmetic and independently reconstructed radio output only. No new inference/test/seed evidence/equivalence or deletion certification. Final retrain reference differs in training stage.')
(out/'certificate.json').write_text(json.dumps(certificate,indent=2)+'\n')
print(json.dumps({k:v for k,v in certificate.items() if k not in ['vectors','cohorts']}))
