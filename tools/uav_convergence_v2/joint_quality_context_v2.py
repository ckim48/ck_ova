"""Second registered screen: forget KL with retained negative context.

The initial screen exposed a utility/divergence tradeoff. This tests a concrete
local-gallery mismatch, not new evaluation thresholds or reporting references.
"""
import argparse
import fcntl
from pathlib import Path
import time
import numpy as np
import joint_quality_control_v1 as first

ROOT,read,write,sha=first.ROOT,first.read,first.write,first.sha
OUT=ROOT/'results/joint_quality_context_v2_20261030'
VARIANTS={'control':first.spec()}
for batch,rho in [(16,None),(16,1.),(32,.5),(32,1.)]:
    for weight in [.25,.5,1.]:
        key=f'context_b{batch}_rho{rho}_rm{weight}'
        VARIANTS[key]=first.spec(batch=batch,rho=rho,retention=weight,threshold=.001)
        VARIANTS[key].update(context=32,forget_weight=3.)
for fw in [1.,10.]:
    key=f'context_b16_rm05_fw{fw}'
    VARIANTS[key]=dict(first.spec(retention=.5,threshold=.001),context=32,forget_weight=fw)
BASE_GRADIENT=first.request_gradient
BASE_VECTORS=first.vectors


def register():
    path=OUT/'protocol.json'
    if path.exists():
        p=read(path);assert p['source_sha256']==sha(Path(__file__));return p
    p=dict(registered_unix=time.time(),source_sha256=sha(Path(__file__)),
        first_source_sha256=sha(Path(first.__file__)),variants=VARIANTS,seed=0,datasets=first.audit.DATASETS,
        intervention='Extend each local forget-query distribution from just that request to a gallery containing its forgotten videos and32 deterministic current retained training videos from the same UAV. Both retrieval directions; evaluate KL only on forget queries. All native captions included. Teacher remains original proxy20; no validation or reporting-reference access.',
        sampling='Include the original retained16 batch, then seeded local retained samples without replacement until32. No currently deleted sample is a context negative.',
        hypothesis='A request-only softmax cannot constrain retrieval against retained distractors. Negative context tests that mismatch; the old proxy age mismatch remains. Fixed coefficients/shared candidates, not a guarantee of true retraining equivalence.',
        accounting='Exact screen only. Instrument every actual student/teacher feature pass and additional score/softmax/backward arithmetic; charge these if physically rerun.',
        selection='Same full evaluation and qualification as first screen; all outcomes retained. Second exploratory stage after inspecting the first screen, not independent confirmation.')
    write(path,p);return p


def vectors(engine,theta,plan,queue,t,fl,kind,ids,proxy,teachers,crossmodal):
    engine.context_pools=engine.joint_base.clip.stream.retained_pools(plan,queue,t)
    return BASE_VECTORS(engine,theta,plan,queue,t,fl,kind,ids,proxy,teachers,crossmodal)


def gradient(engine,theta,fids,rids,step,teacher):
    cfg=first.CURRENT
    if not cfg.get('context'):return BASE_GRADIENT(engine,theta,fids,rids,step,teacher)
    import torch
    import torch.nn.functional as F
    user=engine.sample_owner[int(fids[0])];pool=engine.context_pools[user]
    assert not set(fids)&set(pool)
    remaining=sorted(set(pool)-set(rids));n=min(len(remaining),cfg['context']-len(rids))
    extra=np.random.default_rng(930000+sum(fids)*17+step).choice(remaining,n,replace=False).astype(int).tolist()
    context=list(rids)+extra;ids=list(fids)+context
    fcids=[c for i in fids for c in engine.data.caption_ids[i]]
    cids=[c for i in ids for c in engine.data.caption_ids[i]]
    rcids=engine.data.training_caption_ids(rids,21+step)
    variable=theta.detach().requires_grad_(True)
    im,tx,scale=engine.features(variable,ids,cids)
    oldi,oldt=engine.joint_proxy.features(ids,cids)
    z=scale.exp()*(im@tx.T);target=scale.detach().exp()*(oldi@oldt.T)
    fi=len(fids);fc=len(fcids)
    fk=.5*(F.kl_div(F.log_softmax(z[:fi],1),F.softmax(target[:fi],1),reduction='batchmean')+
           F.kl_div(F.log_softmax(z.T[:fc],1),F.softmax(target.T[:fc],1),reduction='batchmean'))
    ri,rt,scale=engine.features(variable,rids,rcids);oldr=teacher.features(rids,rcids)
    logits=scale.exp()*(ri@rt.T);labels=torch.arange(len(rids),device=theta.device)
    rm=.5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))
    ck=first.quality.infra.lib.adaptive.retrieval_kl(ri,rt,*oldr,scale)
    ff=cfg['forget_weight']*torch.relu(fk-cfg['threshold'])
    g0=torch.autograd.grad(ff+cfg['consistency']*ck,variable,retain_graph=True)[0].detach();g0[-1]=0
    gr=torch.autograd.grad(rm,variable)[0].detach();gr[-1]=0
    g=g0+cfg['retention']*gr
    ops=(len(ids)*len(cids)+len(rids)*len(rcids))*(4*im.shape[1]+32)
    ops+=len(rids)*len(rcids)*(im.shape[1]+16)+2*10240*(len(rids)+len(rcids))+16*(len(theta)-1)+64
    first.quality.infra.lib.infra.charge(engine,fids,int(ops))
    return g,dict(FU_variant='retained_context_KL',forget_loss=float(fk.detach()),RM=float(rm.detach()),CM=float(ck.detach()),
        threshold=cfg['threshold'],RM_weight=cfg['retention'],FU_gradient_norm=float(g.norm()),
        retained_images=rids,context_images=context,extra_control_MAC_equivalents=int(ops))


def install():
    first.OUT=OUT;first.register=register;first.VARIANTS=VARIANTS
    first.vectors=vectors;first.request_gradient=gradient


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','screen']);p.add_argument('--dataset',choices=first.audit.DATASETS)
    a=p.parse_args();install()
    if a.action=='register':register()
    else:
        for ds in [a.dataset] if a.dataset else first.audit.DATASETS:first.screen(ds)
