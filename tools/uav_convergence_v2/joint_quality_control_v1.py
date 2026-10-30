"""Registered training-only joint mR/KL development controls.

All candidates and losing results retained; no reporting reference is a teacher.
Exact screens are not substitutes for physical or independently held-out results.
"""
import argparse
import fcntl
import math
from pathlib import Path
import time

import numpy as np
import forgetting_quality_v1 as audit
import learning_quality_v3 as quality

ROOT,read,write,sha=audit.ROOT,audit.read,audit.write,audit.sha
OUT=ROOT/'results/joint_quality_control_v1_20261030'
CURRENT=None


def spec(**kw):
    return dict(dict(batch=16,rho=None,retention=1.,threshold=.01,consistency=3.,
                     project=False,stop_when_satisfied=False,late_uniform=None),**kw)


VARIANTS={'control':spec()}
for weight in [0.,.25,.5]:
    for threshold in [.01,.001]:
        VARIANTS[f'retain{weight:g}_gate{threshold:g}']=spec(retention=weight,threshold=threshold)
VARIANTS.update({
    'project':spec(project=True),
    'project_tight':spec(project=True,threshold=.001,retention=.5),
    'stop_satisfied':spec(stop_when_satisfied=True),
    'stop_tight':spec(stop_when_satisfied=True,threshold=.001),
    'bound75':spec(rho=.75),
    'bound100':spec(rho=1.),
    'bound25_retain25':spec(rho=.25,retention=.25),
    'bound50_retain50':spec(rho=.5,retention=.5),
    'batch32':spec(batch=32),
    'batch32_bound25':spec(batch=32,rho=.25),
    'batch32_bound50':spec(batch=32,rho=.5),
    'batch32_bound25_retain25':spec(batch=32,rho=.25,retention=.25),
    'uniform_after60':spec(late_uniform=60),
    'uniform_after100':spec(late_uniform=100),
})


def register():
    path=OUT/'protocol.json'
    if path.exists():
        p=read(path)
        assert p['source_sha256']==sha(Path(__file__)),'Preserve frozen study; use a new version for changes.'
        return p
    inherited=read(audit.OUT/'protocol.json')
    p=dict(registered_unix=time.time(),variants=VARIANTS,datasets=audit.DATASETS,seed=0,
        source_sha256=sha(Path(__file__)),input_sha256=inherited['input_sha256'],
        purpose='Find a common setting that improves fixed-reference forget KL while maintaining current retained mR leadership on both datasets. No guaranteed rank, no baseline modification, all outcomes exported.',
        screen='Fresh warm20 to200 FL runs, same45 requests/225 prescribed FU cycles and original schedule, exact aggregation. Faster diagnosis only; no physical feasibility/energy/delay claim. The control must reproduce the previous exact-P-KS final tensor.',
        search='21 predeclared settings shared between datasets. Development evaluation reuses already-inspected seed0 cohorts; not independent test confirmation. Do not select a different setting for each dataset.',
        FL='Current half-random/half-cached-training-loss priority sampler; optional batch32. Bound variants also compute uniform16 gradient and constrain the priority-minus-uniform contribution to rho times its norm. Late-uniform schedules change the sampler at a declared round. Extra executed operations counted, never borrowed reference gradients.',
        FU='3*relu(local forget proxy-KL - threshold) + retention*retainedCE + consistency*request-start-teacher KL. Optional projection removes the preservation-gradient component opposing the forgetting gradient. Optional stop zeros the entire FU update when its local proxy constraint is satisfied. Zero updates still charge executed work; completed prescribed cycles do not certify deletion.',
        teachers='Only original known-cohort retained-only proxy20 and request-start teacher. Reference200, validation queries and evaluation probabilities never enter training. Pre-existing future-forget-cohort assumption is unchanged.',
        scope='Same frozen encoders/adapters and temperature. All deleted videos/native captions. Primary common reference200 stays unchanged. Policy-matched references, if needed for shortlisted settings, are evaluation-only and accompany common KL.',
        qualification='Primary success requires retained mR >= original P-KS on both datasets AND common mean and worst-request KL better than all three unmodified comparators. Also report weaker tradeoff outcomes; failure to qualify is an explicit result.',
        background=[dict(title='SLUG ICML2025',url='https://proceedings.mlr.press/v267/cai25f.html'),
                    dict(title='CLIPErase ACL2025',url='https://aclanthology.org/2025.acl-long.1469/')])
    write(path,p);return p


def set_variant(name):
    global CURRENT
    CURRENT=VARIANTS[name]
    quality.CONFIG=dict(objective='gaft',selection=CURRENT['batch'],equal=False,exact=False)
    quality.ACTIVE=name


def fl_vectors(engine,theta,plan,queue,t,fl):
    from train import update_theta
    base=engine.joint_base;k=21+fl;lr=base.core.lr_at(k)
    pools=base.clip.stream.retained_pools(plan,queue,t);total=sum(map(len,pools))
    result=np.zeros((10,len(theta)-1));records=[]
    for user,pool in enumerate(pools):
        uniform=np.random.default_rng(730000+k*10+user).choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
        if CURRENT['late_uniform'] is not None and k>CURRENT['late_uniform']:
            batch=uniform
        else:
            batch=quality.select_batch(engine,theta,pool,k,user)
        if CURRENT['rho'] is None:
            g,_=engine.gradient(theta,batch,k);beta=1.
        else:
            gu,_=engine.gradient(theta,uniform,k);gh,_=engine.gradient(theta,batch,k)
            delta=gh-gu;beta=min(1.,CURRENT['rho']*float(gu.norm())/max(float(delta.norm()),1e-30))
            g=gu+beta*delta
            assert float((g-gu).norm())<=CURRENT['rho']*float(gu.norm())*(1+1e-5)+1e-8
            quality.infra.lib.infra.charge(engine,uniform,8*len(theta)+16)
        result[user]=(-lr*len(pool)/total*g[:-1]).cpu().numpy()
        records.append(dict(user=user,batch=batch,beta=beta,round=k))
    engine.joint_fl_records=records
    return result


def request_gradient(engine,theta,fids,rids,step,teacher):
    import torch
    import torch.nn.functional as F
    variable=theta.detach().requires_grad_(True)
    cids=[c for i in fids for c in engine.data.caption_ids[i]]
    rcids=engine.data.training_caption_ids(rids,21+step)
    im,tx,scale=engine.features(variable,fids,cids)
    old=engine.joint_proxy.features(fids,cids)
    fk=quality.infra.lib.adaptive.retrieval_kl(im,tx,*old,scale)
    ri,rt,scale=engine.features(variable,rids,rcids);oldr=teacher.features(rids,rcids)
    z=scale.exp()*(ri@rt.T);target=torch.arange(len(rids),device=theta.device)
    rm=.5*(F.cross_entropy(z,target)+F.cross_entropy(z.T,target))
    ck=quality.infra.lib.adaptive.retrieval_kl(ri,rt,*oldr,scale)
    ff=3*torch.relu(fk-CURRENT['threshold'])
    # Keep original gradient accumulation ordering for the no-change control.
    g0=torch.autograd.grad(ff+CURRENT['consistency']*ck,variable,retain_graph=True)[0].detach();g0[-1]=0
    gr=torch.autograd.grad(rm,variable,retain_graph=CURRENT['project'])[0].detach();gr[-1]=0
    g=g0+CURRENT['retention']*gr;removed=0.
    if CURRENT['project']:
        gf=torch.autograd.grad(ff,variable)[0].detach();gf[-1]=0
        gp=g-gf;dot=float(torch.dot(gp,gf));norm=float(torch.dot(gf,gf))
        if dot<0 and norm>1e-24:
            correction=dot/norm*gf;g=g-correction;removed=float(correction.norm())
    stopped=CURRENT['stop_when_satisfied'] and float(fk.detach())<=CURRENT['threshold']
    if stopped:g=torch.zeros_like(g)
    n=len(theta)-1;dim=im.shape[1]
    extra=(len(fids)*len(cids)+len(rids)*len(rcids))*(4*dim+32)
    extra+=len(rids)*len(rcids)*(dim+16)+2*10240*(len(rids)+len(rcids))+16*n+64
    if CURRENT['project']:extra+=6*10240*(len(fids)+len(cids))+8*n
    quality.infra.lib.infra.charge(engine,fids,int(extra))
    return g,dict(FU_variant='joint_quality_control',forget_loss=float(fk.detach()),RM=float(rm.detach()),
        CM=float(ck.detach()),threshold=CURRENT['threshold'],RM_weight=CURRENT['retention'],
        FU_gradient_norm=float(g.norm()),projection_correction_norm=removed,stopped=stopped,
        retained_images=rids,extra_control_MAC_equivalents=int(extra))


def vectors(engine,theta,plan,queue,t,fl,kind,ids,proxy,teachers,crossmodal):
    base=engine.joint_base;engine.joint_proxy=proxy
    l=np.zeros((10,len(theta)-1));u=np.zeros_like(l);stats=[]
    if kind in ['fl','joint']:l=fl_vectors(engine,theta,plan,queue,t,fl)
    pools=base.clip.stream.retained_pools(plan,queue,t)
    total=sum(len(queue.items[i]['images']) for i in ids)
    for rid in ids:
        req=queue.items[rid];user=req['user'];step=req['steps'];fids=req['images']
        if rid not in teachers:teachers[rid]=base.core.Teacher(engine,theta)
        seed=910000+req['burst']*1000+user*10+step
        rids=np.random.default_rng(seed).choice(pools[user],min(16,len(pools[user])),replace=False).astype(int).tolist()
        g,meta=request_gradient(engine,theta,fids,rids,step,teachers[rid])
        u[user]+=(-base.core.lr_at(21+fl)*len(fids)/total*g[:-1]).cpu().numpy()
        stats.append(dict(request=rid,step=step+1,FU_learning_rate=base.core.lr_at(21+fl),**meta))
    return l,u,stats


def evaluate(engine,theta,plan,refprob,requests,dest):
    summary,_=audit.store_eval(engine,theta,plan,dest,dict(common=refprob))
    per=np.load(dest/'per_video.npz');r=[]
    for req in requests:
        idx=[plan['gallery_images'].index(i) for i in req['images']]
        r.append(dict(request=req['id'],KL=float(per['common_KL'][idx].mean())))
    write(dest/'request_kl.json',r)
    return dict(retained_mR=100*summary['retained']['mR'],retained_loss=summary['retained']['loss'],
                common_KL=summary['forget']['common_KL'],worst_KL=max(x['KL'] for x in r),
                forget_mR=100*summary['forget']['mR'])


def screen(dataset):
    import torch
    register();assert torch.cuda.is_available();torch.set_num_threads(1)
    base,data,parent,plan,payload,anchor=audit.runtime(dataset)
    from train import update_theta
    original=read(anchor/'service_completion.json')['evaluation']
    reference=base.core.load_theta(anchor.parent/'reference200/final.pt','cuda')
    warm=base.core.load_theta(anchor.parent/'initial20/final.pt','cuda')
    proxy_theta=base.core.load_theta(anchor.parent/'proxy20/final.pt','cuda')
    requests=read(anchor.parent/'requests.json')
    txs=[read(f) for f in sorted((anchor/'transactions').glob('*.json'))]
    with (ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name in VARIANTS:
            dest=OUT/'screen'/dataset.lower()/name
            if (dest/'finished.json').exists():continue
            assert not (dest/'started.json').exists(),'Preserve interrupted attempt.'
            write(dest/'started.json',dict(unix=time.time(),configuration=VARIANTS[name]))
            set_variant(name);engine=quality.engine_wrapper(parent)(data,plan,payload);engine.joint_base=base
            engine.set_teacher(warm);theta=warm.clone();proxy=base.core.Teacher(engine,proxy_theta)
            _,refprob=audit.per_video(engine,reference,plan)
            queue=base.RoundQueue(requests);teachers={};fl=0;records=[];t0=time.time()
            for tx in txs:
                queue.sync(fl,tx['start_s'],0.);engine.start_work()
                l,u,stats=vectors(engine,theta,plan,queue,tx['start_s'],fl,tx['kind'],tx['FU_request_ids'],proxy,teachers,False)
                theta=update_theta(theta,(l+u).sum(0));work=engine.finish_work()
                queue.commit(tx['FU_request_ids'],tx['start_s'],tx['end_s']);fl+=int(tx['kind'] in ['fl','joint'])
                records.append(dict(sequence=tx['sequence'],FL_round=20+fl,local=stats,
                    device_macs=work['device_macs'],FL_norm=float(np.linalg.norm(l)),FU_norm=float(np.linalg.norm(u))))
            assert fl==180 and sum(len(r['local']) for r in records)==225
            if name=='control':
                prev=audit.OUT/dataset.lower()/'gate_select16/exact_fu/final.pt'
                assert torch.equal(theta,base.core.load_theta(prev,engine.device)),'No-change exact control mismatch'
            base.core.save_theta(dest/'final.pt',theta,200);write(dest/'work.json',records)
            metrics=evaluate(engine,theta,plan,refprob,requests,dest)
            result=dict(dataset=dataset,variant=name,wall_s=time.time()-t0,configuration=VARIANTS[name],
                FL_rounds=200,FU_steps=225,exact_control_reproduced=name=='control',**metrics)
            write(dest/'finished.json',result);print(result,flush=True)
    write(OUT/'screen'/dataset.lower()/'finished.json',dict(passed=True,candidates=len(VARIANTS)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','screen']);p.add_argument('--dataset',choices=audit.DATASETS)
    a=p.parse_args()
    if a.action=='register':register()
    else:
        for ds in [a.dataset] if a.dataset else audit.DATASETS:screen(ds)
