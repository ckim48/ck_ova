"""Finite training-only FU retention refinement; retain all outcomes and references."""
import argparse
import fcntl
from pathlib import Path
import time
import numpy as np
import joint_quality_control_v1 as first
import joint_quality_audit_v1 as prior_audit

ROOT,read,write,sha=first.ROOT,first.read,first.write,first.sha
OUT=ROOT/'results/forgetting_retention_refine_v1_20261030'
VARIANTS={'control':first.spec(),'base100':first.spec(late_uniform=100)}
for weight in [.95,.9,.85,.75,.5]:
    VARIANTS[f'retain{weight:g}']=first.spec(late_uniform=100,retention=weight)


def register():
    file=OUT/'protocol.json'
    if file.exists():
        p=read(file);assert p['source_sha256']==sha(Path(__file__));return p
    p=dict(registered_unix=time.time(),source_sha256=sha(Path(__file__)),
        inherited_source_sha256=sha(Path(first.__file__)),variants=VARIANTS,seed=0,
        datasets=first.audit.DATASETS,
        purpose='Investigate actual CapERA matched-KL improvement with retained-mR leadership preserved on both datasets. Not an instruction to force a winning curve.',
        design='Seven declared configurations, same coefficient on both datasets. Original exact control and late-uniform100 control are included. Seed0 adaptive development after prior screens, not independent confirmation.',
        training='Only the retained-CE coefficient inside the existing FU gradient changes. Priority16 through FL100 then uniform16. Keep proxy20, original gate0.01, consistency3, same rounds200/requests45/prescribed cycles225, native captions and fixed temperature. Reporting references/probabilities are never training teachers.',
        selection='Prefer a shared setting maintaining at least original P-KS retained mR on both datasets and improving mean and worst-request matched KL. Validate up to two shortlisted settings under actual wireless caps before adoption; report every result. No condition guarantees leadership.',
        reference='All late-uniform variants have the same FL-only policy, so use the previously trained uniform_after100 retained-only reference and exact-no-FU control. The common reference remains unchanged and is always reported. Original control uses the original priority matched reference.',
        scope='Final matched KL is a secondary retraining-similarity metric. Do not equate a winning endpoint with a causal FU effect or common-reference lead. Carry paired no-FU results forward.',
        control_hashes={str((prior_audit.OUT/ds.lower()/name/'final.pt').relative_to(ROOT)):sha(prior_audit.OUT/ds.lower()/name/'final.pt')
            for ds in first.audit.DATASETS for name in ['uniform_after100_reference','uniform_after100_exact_no_fu']})
    write(file,p);return p


def install():
    first.OUT=OUT;first.register=register;first.VARIANTS=VARIANTS


def evaluate(ds,mode):
    import torch
    base,data,parent,plan,payload,anchor=first.audit.runtime(ds)
    first.set_variant('base100');engine=first.quality.engine_wrapper(parent)(data,plan,payload)
    requests=read(anchor.parent/'requests.json')
    if mode=='screen':
        sources=[(v,OUT/'screen'/ds.lower()/v/'final.pt') for v in VARIANTS]
    else:
        sources=[(p.parts[-6],p) for p in sorted((OUT/'physical/retention').glob('*/cases/baseline/*/adaptive_retrieval_fu/final.pt')) if p.parent.parent.name==ds.lower()+'_seed0']
    common=base.core.load_theta(anchor.parent/'reference200/final.pt',engine.device)
    matched=base.core.load_theta(prior_audit.OUT/ds.lower()/'uniform_after100_reference/final.pt',engine.device)
    _,cp=first.audit.per_video(engine,common,plan);_,mp=first.audit.per_video(engine,matched,plan)
    rows=[]
    for variant,path in sources:
        theta=base.core.load_theta(path,engine.device)
        mprob=mp
        if variant=='control':
            ref=base.core.load_theta(first.audit.OUT/ds.lower()/'references/priority/final.pt',engine.device)
            _,mprob=first.audit.per_video(engine,ref,plan)
        if mode=='screen' and variant=='base100':
            original=ROOT/'results/joint_quality_control_v1_20261030/screen'/ds.lower()/'uniform_after100/final.pt'
            assert torch.equal(theta,base.core.load_theta(original,engine.device))
        dest=OUT/'evaluation'/mode/ds.lower()/variant
        metrics,_=first.audit.store_eval(engine,theta,plan,dest,dict(common=cp,matched=mprob))
        per=np.load(dest/'per_video.npz');request_rows=[]
        for req in requests:
            idx=[plan['gallery_images'].index(i) for i in req['images']]
            request_rows.append(dict(request=req['id'],common_KL=float(per['common_KL'][idx].mean()),matched_KL=float(per['matched_KL'][idx].mean())))
        write(dest/'request_kl.json',request_rows)
        row=dict(dataset=ds,variant=variant,mode=mode,configuration=VARIANTS[variant],
            retained_mR=100*metrics['retained']['mR'],retained_loss=metrics['retained']['loss'],
            forget_mR=100*metrics['forget']['mR'],common_KL=metrics['forget']['common_KL'],matched_KL=metrics['forget']['matched_KL'],
            worst_common_KL=max(r['common_KL'] for r in request_rows),worst_matched_KL=max(r['matched_KL'] for r in request_rows),
            source=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),matched_reference='priority' if variant=='control' else 'uniform_after100')
        if mode=='physical':
            arm=path.parent
            for filename in ['plan.json','requests.json','reference200/final.pt','initial20/final.pt']:
                assert sha(arm.parent/filename)==sha(anchor.parent/filename)
            verification=read(arm/'verification.json');cost=read(arm/'energy_and_delay.json')
            assert verification['passed'] and verification['FL_rounds']==200 and verification['FU_steps']==225
            assert cost['network_cap_overrun_transactions']==0
            for txpath in (arm/'transactions').glob('*.json'):
                r=read(txpath)['radio'];assert r['max_power_w']<=.32+1e-6 and r['energy_bound_j']<=.1+1e-6
                if r.get('access')=='zero_update_silence':assert r['conditional_rmse']==0
                else:assert r['error_bound']<=r['epsilon']*(1+1e-5)+1e-9
            assert abs(row['common_KL']-cost['forget_KL'])<1e-6
            row.update(energy_kJ=cost['total_energy_j']/1000,active_delay_s=cost['delay_s'],verification=verification)
        write(dest/'summary.json',row);rows.append(row);print({k:v for k,v in row.items() if k not in ['configuration','verification','source','checkpoint_sha256']},flush=True)
    write(OUT/'evaluation'/mode/ds.lower()/'summary.json',rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['screen','physical','evaluate']);parser.add_argument('--variant');parser.add_argument('--mode',choices=['screen','physical'],default='physical')
    args=parser.parse_args();install();register()
    if args.action=='physical':
        import joint_quality_physical_v2 as physical
        assert args.variant in VARIANTS
        physical.OUT=OUT/'physical';physical.run('retention',args.variant)
    else:
        import torch
        torch.set_num_threads(1);assert torch.cuda.is_available()
        if args.action=='screen':
            for ds in first.audit.DATASETS:first.screen(ds)
        with (ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for ds in first.audit.DATASETS:evaluate(ds,'screen' if args.action=='screen' else args.mode)
