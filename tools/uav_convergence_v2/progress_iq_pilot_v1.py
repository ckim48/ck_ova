"""Learning-anchored FU progress control; disclosed follow-up to failed RMS pilot."""
import argparse
from pathlib import Path
import time
import numpy as np
import adaptive_iq_pilot_v2 as earlier

ROOT,read,write,sha=earlier.ROOT,earlier.read,earlier.write,earlier.sha
learning,audit,previous=earlier.learning,earlier.audit,earlier.previous
OUT=ROOT/'results/progress_iq_pilot_v1_20261030'
SETTINGS={f'r{r}_p{p:g}':dict(retention=r,progress=p) for r in [1,0] for p in [.25,1.]}
MODES=['combined','combined_local','iq_progress']
CURRENT=None


def fuse(l,u,rho):
    l,u=np.asarray(l,float),np.asarray(u,float)
    a,b,d=float(l@l),float(u@u),float(l@u)
    radius=np.sqrt(a)+np.sqrt(b)
    target=rho*b
    if b==0: result=l.copy()
    elif a==0: result=rho*u
    elif d>=target: result=l.copy()
    elif d>=0:
        need=(target-d)/b
        limit=(-d+np.sqrt(max(0.,d*d+b*(radius*radius-a))))/b
        result=l+min(need,limit)*u
    else:
        perpendicular=u-(d/a)*l
        norm=float(perpendicular@perpendicular)
        if norm<=1e-24*max(a,b):result=l.copy()
        else:
            need=(target-d)/norm
            limit=np.sqrt(max(0.,radius*radius-a)/norm)
            result=l+min(need,limit)*perpendicular
    achieved=float(u@result)
    assert np.linalg.norm(result)<=radius*(1+1e-10)+1e-15
    assert float(l@result)>=a*(1-1e-10)-1e-18
    return result,dict(FL_progress=float(l@result),FL_target=a,FU_progress=achieved,FU_target=target,
        FU_target_met=achieved>=target*(1-1e-9)-1e-18,cosine=d/max(np.sqrt(a*b),1e-30),
        update_norm=float(np.linalg.norm(result)),radius=float(radius),
        correction_norm=float(np.linalg.norm(result-l-u)))


def register():
    path=OUT/'protocol.json'
    if path.exists():
        p=read(path)
        assert sha(Path(__file__))==p['source_sha256']
        return p
    old=read(earlier.OUT/'selection.json')
    assert not any(r['joint_winner'] for r in old['rankings'])
    p=dict(registered_unix=time.time(),source_sha256=sha(Path(__file__)),settings=SETTINGS,modes=MODES,
        motivation='All four RMS candidates failed the joint mR/common-KL/matched-KL criterion. Preserve them and defer their planned physical confirmation because no qualifying candidate emerged. This is a new disclosed development hypothesis, not a hidden extension of the original grid.',
        learning='Same manuscript cosine FU and uniform16 FL. Retained CE coefficient in FU is either1 (original) or0 (avoid duplicated retained training already performed by FL). Both comparators use exactly the same selected loss. No proxy or evaluation reference is a training teacher.',
        rule='Minimize squared distance from the FL-only global step while preserving its first-order retained progress and requiring rho times the FU-only first-order progress. Restrict norm to ||L||+||U||. Analytic two-halfspace solution; if radius prevents the target, log the unmet FU progress explicitly. Estimates do not certify true loss decrease.',
        baselines='combined transmits L+rho*U with rho known before transmission. combined_local applies the same progress rule independently at every UAV, then combines. iq_progress applies it to separately recovered GLOBAL updates. Equal data/steps/LR and actual waveform costs in any subsequent physical run.',
        grid='4 settings x3 methods x2 datasets, seed0 exact pilot. Shared retention/rho choice across datasets. All outcomes retained. No RF/energy claims for the pilot.',
        evaluation='Both fixed common and uniform/equal-policy matched retrains; all native forget/retained queries. FU terms are also compared to a paired FL-only control before interpreting forgetting quality.',
        selection='Same worst-normalized-margin score as RMS pilot, using mR and common/matched KL against both Combined baselines on both datasets. No guaranteed winner. Any later selection/amendment must be explicit.',
        theoretical_scope='A nonlinear head-side control changes the ideal update. The earlier linear-sum C8 distortion guarantee does not apply after this rule.')
    rng=np.random.default_rng(41)
    for _ in range(500):
        l,u=rng.normal(size=(2,128));fuse(l,u,.25);fuse(l,u,1.)
    write(path,p);return p


def gradient(engine,theta,fids,rids,step,teacher):
    import torch
    x=theta.detach().requires_grad_(True)
    fm,rm,cm=previous.manuscript_terms(engine,x,fids,rids,step,teacher)
    g=torch.autograd.grad(3*fm+CURRENT['retention']*rm+3*cm,x)[0].detach();g[-1]=0
    nc=sum(len(engine.data.caption_ids[i]) for i in fids)
    extra=int((len(fids)*nc+len(rids)**2)*(4*512+32)+2*10240*len(rids)*2+16*(len(theta)-1)+64)
    learning.quality.infra.lib.infra.charge(engine,fids,extra)
    return g,dict(FU_variant='progress_cosine',forget_loss=float(fm.detach()),RM=float(rm.detach()),
        CM=float(cm.detach()),RM_weight=CURRENT['retention'],FU_gradient_norm=float(g.norm()),
        retained_images=rids,extra_control_MAC_equivalents=extra)


def run():
    global CURRENT
    import torch
    register();earlier.audit.runtime('CapERA')
    from train import update_theta
    earlier.original.configure();learning.request_gradient=gradient
    torch.set_num_threads(1);assert torch.cuda.is_available()
    for ds in ['CapERA','AirScape']:
        base,data,parent,plan,payload,anchor=audit.runtime(ds)
        transactions=[read(p) for p in sorted((anchor/'transactions').glob('*.json'))]
        requests=read(anchor.parent/'requests.json')
        for setting,CURRENT in SETTINGS.items():
            for mode in MODES:
                dest=OUT/'runs'/ds.lower()/setting/mode
                if (dest/'finished.json').exists():continue
                assert not (dest/'started.json').exists()
                write(dest/'started.json',dict(unix=time.time()));start=time.time()
                engine=learning.quality.engine_wrapper(parent)(data,plan,payload);engine.joint_base=base
                theta=base.core.load_theta(anchor.parent/'initial20/final.pt',engine.device);engine.set_teacher(theta)
                queue,teachers=base.RoundQueue(requests),{};fl=0;rows=[]
                for tx in transactions:
                    queue.sync(fl,tx['start_s'],0.);engine.start_work()
                    l,u,local=learning.vectors(engine,theta,plan,queue,tx['start_s'],fl,tx['kind'],tx['FU_request_ids'],None,teachers,False)
                    if mode=='combined':delta=(l+CURRENT['progress']*u).sum(0);control=[]
                    elif mode=='combined_local':
                        pairs=[fuse(a,b,CURRENT['progress']) for a,b in zip(l,u)]
                        delta=sum(x[0] for x in pairs);control=[x[1] for x in pairs]
                    else:
                        delta,info=fuse(l.sum(0),u.sum(0),CURRENT['progress']);control=[info]
                    theta=update_theta(theta,delta);engine.finish_work()
                    queue.commit(tx['FU_request_ids'],tx['start_s'],tx['end_s']);fl+=1
                    rows.append(dict(round=20+fl,control=control,local=local))
                assert fl==180 and sum(len(r['local']) for r in rows)==225
                base.core.save_theta(dest/'final.pt',theta,200)
                refs={key:audit.per_video(engine,base.core.load_theta(path,engine.device),plan)[1] for key,path in dict(
                    common=anchor.parent/'reference200/final.pt',matched=audit.OUT/ds.lower()/'references/equal/final.pt').items()}
                metrics,_=audit.store_eval(engine,theta,plan,dest,refs)
                result=dict(dataset=ds,setting=setting,mode=mode,mR=100*metrics['retained']['mR'],loss=metrics['retained']['loss'],
                    common_KL=metrics['forget']['common_KL'],matched_KL=metrics['forget']['matched_KL'],wall_s=time.time()-start,
                    unmet_targets=sum(not c['FU_target_met'] for row in rows for c in row['control']))
                write(dest/'work.json',rows);write(dest/'finished.json',result);print(result,flush=True)
    select()


def select():
    rows=[read(p) for p in sorted((OUT/'runs').glob('*/*/*/finished.json'))];assert len(rows)==24
    ranks=[]
    for setting in SETTINGS:
        margins=[]
        for ds in ['CapERA','AirScape']:
            a=next(r for r in rows if r['setting']==setting and r['dataset']==ds and r['mode']=='iq_progress')
            for mode in ['combined','combined_local']:
                b=next(r for r in rows if r['setting']==setting and r['dataset']==ds and r['mode']==mode)
                margins.extend([(a['mR']-b['mR'])/.1,(1-a['common_KL']/b['common_KL'])/.01,(1-a['matched_KL']/b['matched_KL'])/.01])
        ranks.append(dict(setting=setting,worst_margin=min(margins),joint_winner=min(margins)>=-1e-8 and max(margins)>1e-8))
    ranks.sort(key=lambda r:(-r['worst_margin'],r['setting']))
    write(OUT/'selection.json',dict(selected=ranks[0]['setting'],rankings=ranks,results=rows))
    print('SELECTION',ranks,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);a=p.parse_args()
    {'register':register,'run':run}[a.action]()
