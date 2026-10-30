"""Fixed-cohort physical evaluation and causal controls for the shared FL schedule.

References are evaluation only. No-FU is exact-aggregation bookkeeping, not deletion.
"""
import fcntl
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import joint_quality_control_v1 as first

ROOT,read,write,sha=first.ROOT,first.read,first.write,first.sha
OUT=ROOT/'results/joint_quality_audit_v1_20261030'
PHYSICAL=ROOT/'results/joint_quality_physical_v2_20261030'
VARIANT='uniform_after100'


def run(ds):
    import torch
    base,data,parent,plan,payload,anchor=first.audit.runtime(ds)
    from train import update_theta
    first.set_variant(VARIANT)
    engine=first.quality.engine_wrapper(parent)(data,plan,payload);engine.joint_base=base
    common=base.core.load_theta(anchor.parent/'reference200/final.pt',engine.device)
    _,common_prob=first.audit.per_video(engine,common,plan)
    requests=read(anchor.parent/'requests.json')
    root=OUT/ds.lower()
    physical=[]
    for arm in sorted(PHYSICAL.glob('*/ */cases/baseline'.replace(' ',''))):
        source=arm/(ds.lower()+'_seed0')/'adaptive_retrieval_fu'
        if not (source/'final.pt').exists():continue
        stage,variant=arm.parts[-4:-2]
        # New physical runs must use the same raw cohorts and fixed reference.
        for filename in ['plan.json','requests.json','reference200/final.pt','initial20/final.pt']:
            assert sha(source.parent/filename)==sha(anchor.parent/filename),(source,filename)
        dest=root/'physical'/stage/variant
        theta=base.core.load_theta(source/'final.pt',engine.device)
        metrics=first.evaluate(engine,theta,plan,common_prob,requests,dest)
        verification=read(source/'verification.json');cost=read(source/'energy_and_delay.json')
        assert verification['passed'] and verification['FL_rounds']==200 and verification['FU_steps']==225
        assert cost['network_cap_overrun_transactions']==0
        assert abs(metrics['retained_mR']-100*cost['mR'])<2e-5
        assert abs(metrics['common_KL']-cost['forget_KL'])<1e-6
        record=dict(dataset=ds,stage=stage,variant=variant,source=str(source.relative_to(ROOT)),
            checkpoint_sha256=sha(source/'final.pt'),verification=verification,cost=cost,**metrics)
        write(dest/'summary.json',record);physical.append(record)
    # A policy-matched reference does not replace the unchanged primary reference.
    refdir=root/'uniform_after100_reference'
    if not (refdir/'finished.json').exists():
        assert not (refdir/'started.json').exists(),'Preserve interrupted controls.'
        write(refdir/'started.json',dict(unix=time.time()))
        records=read((anchor.parent/'reference200').resolve().parent/'reference_records.json')
        engine=first.quality.engine_wrapper(parent)(data,plan,payload);engine.joint_base=base
        theta=engine.theta0.clone();deleted=set(plan['forgotten_images'])
        queue=SimpleNamespace(arrived=lambda t:[dict(images=sorted(deleted))])
        work=[]
        for k in range(1,201):
            engine.start_work()
            if k<=20:
                first.quality.CONFIG['selection']=0
                assert all(not set(b)&deleted for b in records[k-1]['batches'])
                engine.plan['learning_rate']=base.core.lr_at(k)
                l,_,_=engine.updates(theta,records[k-1],k,'retrain',[[] for _ in range(10)])
            else:
                first.set_variant(VARIANT)
                l=first.fl_vectors(engine,theta,plan,queue,0.,k-21)
                assert all(not set(r['batch'])&deleted for r in engine.joint_fl_records)
            theta=update_theta(theta,l.sum(0));work.append(engine.finish_work())
        base.core.save_theta(refdir/'final.pt',theta,200);write(refdir/'work.json',work)
        first.audit.store_eval(engine,theta,plan,refdir,dict(common=common_prob))
        write(refdir/'finished.json',dict(rounds=200,checkpoint_sha256=sha(refdir/'final.pt'),evaluation_only=True))
    ref=base.core.load_theta(refdir/'final.pt',engine.device)
    _,matched_prob=first.audit.per_video(engine,ref,plan)
    # Paired no-FU replay uses identical arrivals and FL seeds/rounds to exact-FU.
    nofu=root/'uniform_after100_exact_no_fu'
    if not (nofu/'finished.json').exists():
        assert not (nofu/'started.json').exists(),'Preserve interrupted controls.'
        write(nofu/'started.json',dict(unix=time.time()))
        first.set_variant(VARIANT)
        engine=first.quality.engine_wrapper(parent)(data,plan,payload);engine.joint_base=base
        theta=base.core.load_theta(anchor.parent/'initial20/final.pt',engine.device);engine.set_teacher(theta)
        queue=base.RoundQueue(requests);fl=0;work=[]
        for path in sorted((anchor/'transactions').glob('*.json')):
            tx=read(path);queue.sync(fl,tx['start_s'],0.);engine.start_work()
            l=first.fl_vectors(engine,theta,plan,queue,tx['start_s'],fl)
            theta=update_theta(theta,l.sum(0));work.append(engine.finish_work())
            queue.commit(tx['FU_request_ids'],tx['start_s'],tx['end_s']);fl+=1
        assert fl==180
        base.core.save_theta(nofu/'final.pt',theta,200);write(nofu/'work.json',work)
        write(nofu/'finished.json',dict(rounds=200,FU_steps=0,request_commits='Bookkeeping only; no deletion claim.',checkpoint_sha256=sha(nofu/'final.pt')))
    conditions=dict(exact_no_fu=nofu/'final.pt',
        exact_fu=first.OUT/'screen'/ds.lower()/VARIANT/'final.pt',
        physical_fu=PHYSICAL/'first'/VARIANT/'cases/baseline'/(ds.lower()+'_seed0')/'adaptive_retrieval_fu/final.pt')
    diagnostics={}
    for name,path in conditions.items():
        dest=root/'paired'/name;theta=base.core.load_theta(path,engine.device)
        metrics,_=first.audit.store_eval(engine,theta,plan,dest,dict(common=common_prob,matched=matched_prob))
        values=np.load(dest/'per_video.npz');rows=[]
        for req in requests:
            idx=[plan['gallery_images'].index(i) for i in req['images']]
            rows.append(dict(request=req['id'],common_KL=float(values['common_KL'][idx].mean()),matched_KL=float(values['matched_KL'][idx].mean())))
        write(dest/'request_kl.json',rows)
        diagnostics[name]=dict(metrics=metrics,worst_common_KL=max(r['common_KL'] for r in rows),
            worst_matched_KL=max(r['matched_KL'] for r in rows),source=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path))
    result=dict(dataset=ds,physical=physical,paired=diagnostics)
    write(root/'summary.json',result)
    print(dict(dataset=ds,paired={k:v['metrics']['forget'] for k,v in diagnostics.items()}),flush=True)


if __name__=='__main__':
    import torch
    torch.set_num_threads(1);assert torch.cuda.is_available()
    protocol=dict(registered_unix=time.time(),source_sha256=sha(Path(__file__)),variant=VARIANT,
        design='Seed0 exploratory follow-up of an adaptively selected shared setting; not independent confirmation.',
        primary='Same fixed common retained-only200 reference and full gallery as original four methods.',
        secondary='Fresh policy-matched retained-only200 retrain. No reference model/probability enters candidate training.',
        causal='Exact-FU vs exact-no-FU using identical FL schedule, arrivals, seeds and warm20. Physical-FU vs exact-no-FU is not a pure FU intervention.',
        policy='Priority16 through round100, uniform16 for101..200; unchanged original gated FU objective and parallel transport.',
        sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(first.__file__),Path(__file__)]})
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')['source_sha256']==protocol['source_sha256']
    else:write(OUT/'protocol.json',protocol)
    with (ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for ds in first.audit.DATASETS:run(ds)
