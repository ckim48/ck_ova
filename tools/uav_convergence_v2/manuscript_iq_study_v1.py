"""Paired development test of the supplied I/Q methodology against frozen P-KS."""
import argparse
import inspect
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import learning_selection_fix_v2 as selection
import proposed_pks_report as published
import manuscript_iq_radio_v1 as radio

study=selection.study
ROOT,read,write,sha=study.ROOT,study.read,study.write,study.sha
OUT=ROOT/'results/manuscript_iq_v1_20261029'
MANUSCRIPT=Path('/home/wnlab/.codex/attachments/ba103a3b-580b-4327-8158-ecaa7ab24af2/Pasted text.txt')
VARIANTS={'iq_only':'P-KS + manuscript I/Q','paper_iq':'Manuscript I/Q + manuscript learning'}
ACTIVE=None
CURRENT_ELL=None
ORIGINAL_VECTORS=None


def register():
    p=OUT/'protocol.json'
    if p.exists():
        result=read(p)
        for k,v in result['source_sha256'].items():assert sha(ROOT/k)==v,k
        return result
    sources={str(Path(f).relative_to(ROOT)):sha(Path(f)) for f in [__file__,radio.__file__]}
    old={ds:str(published.source('baseline',ds,'gate_select16').relative_to(ROOT)) for ds in published.DATASETS}
    inputs={}
    for arm in old.values():
        for f in ['plan.json','requests.json','initial20/final.pt','reference200/final.pt','feature_cache/features.pt']:
            path=(ROOT/arm).parent/f;inputs[str(path.relative_to(ROOT))]=sha(path)
    result=dict(registered_unix=time.time(),variants=VARIANTS,datasets=published.DATASETS,seed=0,
        manuscript_sha256=sha(MANUSCRIPT),original_Proposed=old,source_sha256=sources,input_sha256=inputs,
        fixed='Same native data, cached features, warm20, requests, 200 FL rounds, 45x5 FU steps, geometry, noise-normalized channels, power0.32W, networkRF0.1J, and reference200. No outcome-dependent tuning.',
        learning='iq_only: unchanged corrected P-KS sampler, weights and gated retrieval loss. paper_iq: uniform FL16, mini-batch-size FL weights (1/10), forget-size FU weights, 3 positive-pair cosine +1 retained CE +3 embedding-softmax teacher KL. Request teacher frozen at first service, original paper-equation implementation reused.',
        unspecified_values='Original weights F/R/C=3/1/3; zeta_L=zeta_U=0.2; floors1e-8; original expansion grid1..16,20,24,32,40,48,56,64. 2*delta_t fixed to half the task distortion of uniform component-C8 error allocation. Values chosen before candidate results, not supplied as numerical constants in manuscript.',
        curvature='At each transmitted update, two deterministic Rademacher HVP probes per active objective, one supported on visual and one on textual coordinates. Block row-norm estimates include cross-modality products; Frobenius scaling and Cauchy row-sum scaling. Empirical surrogate at current model, NOT a certified upper bound over all Gaussian perturbations.',
        solver='Exact phase inversion; independent I/Q noise; joint I+Q block power. SOCP transceiver, smallest feasible fixed-x symbol count, fixed-x enumeration of ten heads. Deterministic first feasible head at max expansion, at most8 BCD iterations; local stationary solution, no global-optimum claim.',
        accounting='Primary comparison retains original communication + computation + hovering including broadcasts and circuits for both arms; also export manuscript compute+uplinkPA+compute/uplink-hover scope separately. Curvature extra feature/teacher passes and conservative HVP MAC estimate charged. Optimizer wall time recorded but not converted to UAV hardware energy.',
        limitations='Single paired development seed. Manuscript does not specify curvature estimator/delta/initialization; this is a disclosed operationalization. Existing table inside attachment is not evidence for the new I/Q implementation. No deletion guarantee or independent test claim.')
    write(p,result);return result


def symmetric_ce(im,tx,scale):
    import torch
    import torch.nn.functional as F
    z=scale.exp()*(im@tx.T);target=torch.arange(len(im),device=im.device)
    return .5*(F.cross_entropy(z,target)+F.cross_entropy(z.T,target))


def fu_loss(engine,theta,fids,rids,step,teacher):
    import torch
    import torch.nn.functional as F
    fcids=[c for i in fids for c in engine.data.caption_ids[i]]
    rcids=engine.data.training_caption_ids(rids,21+step)
    im,tx,scale=engine.features(theta,fids,fcids)
    ri,rt,scale=engine.features(theta,rids,rcids)
    old=teacher.features(rids,rcids)
    rm=symmetric_ce(ri,rt,scale)
    if ACTIVE=='iq_only':
        oldf=engine.adaptive_proxy.features(fids,fcids)
        kl=study.infra.lib.adaptive.retrieval_kl
        return 3*torch.relu(kl(im,tx,*oldf,scale)-.01)+rm+3*kl(ri,rt,*old,scale)
    owner=torch.tensor([j for j,i in enumerate(fids) for _ in engine.data.caption_ids[i]],device=theta.device)
    fm=(im[owner]*tx).sum(-1).mean()
    cm=F.kl_div(F.log_softmax(ri,dim=-1),F.softmax(old[0],dim=-1),reduction='batchmean')
    cm+=F.kl_div(F.log_softmax(rt,dim=-1),F.softmax(old[1],dim=-1),reduction='batchmean')
    return 3*fm+rm+3*cm


def curvature(value,theta,seed):
    import torch
    d=theta.numel()-1;half=d//2;block=128
    grad=torch.autograd.grad(value,theta,create_graph=True)[0]
    result=np.zeros(d//block);rng=np.random.default_rng(seed)
    for start in [0,half]:
        direction=torch.zeros_like(theta)
        direction[start:start+half]=torch.as_tensor(rng.choice([-1.,1.],half)/np.sqrt(half),device=theta.device,dtype=theta.dtype)
        hv=torch.autograd.grad((grad*direction).sum(),theta,retain_graph=True)[0][:-1]
        result+=hv.detach().reshape(-1,block).norm(dim=1).cpu().numpy()*np.sqrt(half)*np.sqrt(half/block)
    assert np.isfinite(result).all()
    return np.maximum(result,1e-12)


def complete_vectors(engine,theta,plan,queue,t,fl,kind,ids,teachers,l,u,stats):
    global CURRENT_ELL
    import torch
    # Rebuild only objectives on batches actually selected above. No validation data.
    variable=theta.detach().requires_grad_(True);values=[];cost=[0]*10
    if engine.iq_batches:
        sizes=[len(set(p)-{i for r in queue.items.values() if r['arrival_s'] is not None and r['arrival_s']<=t+1e-10 for i in r['images']}) for p in plan['partitions']]
        objective=0
        for user,batch,k in engine.iq_batches:
            im,tx,scale=engine.features(variable,batch,engine.data.training_caption_ids(batch,k))
            weight=sizes[user]/sum(sizes) if ACTIVE=='iq_only' else len(batch)/sum(len(b) for _,b,_ in engine.iq_batches)
            objective=objective+weight*symmetric_ce(im,tx,scale)
            cost[user]+=int(12*10240*2*len(batch)+8*len(batch)**2*512)
        values.append(curvature(objective,variable,881000+fl))
    if ids:
        objective=0;total=sum(len(queue.items[r]['images']) for r in ids)
        for row in stats:
            req=queue.items[row['request']];fids=req['images'];rids=row['retained_images']
            objective=objective+len(fids)/total*fu_loss(engine,variable,fids,rids,req['steps'],teachers[row['request']])
            nc=sum(len(engine.data.caption_ids[i]) for i in fids)
            cost[req['user']]+=int(12*10240*(len(fids)+nc+2*len(rids))+8*(len(fids)*nc+len(rids)**2)*512)
        values.append(curvature(objective,variable,992000+fl))
    CURRENT_ELL=np.stack(values) if values else np.ones((1,l.shape[1]//128))
    for user,batch,k in engine.iq_batches:
        study.infra.lib.infra.charge(engine,batch,cost[user]);cost[user]=0
    for rid in ids:
        req=queue.items[rid];user=req['user']
        if cost[user]:study.infra.lib.infra.charge(engine,req['images'],cost[user]);cost[user]=0
    engine.iq_curvature= dict(method='two modality-supported HVP probes per active task',certified=False,
        block_curvature=CURRENT_ELL.tolist(),active_tasks=len(values),hvp_calls=2*len(values))
    return l,u,stats


def vectors(base,objective):
    if ACTIVE=='iq_only':fn=ORIGINAL_VECTORS(base,objective)
    else:
        fn=base.control.synchronized_vectors(objective)
        if objective=='cliperase':
            fn=study.infra.cached_clone(fn,[('len(pool)/total','1/10',1)],{})
    if objective!='cliperase':return fn
    source=inspect.getsource(fn)
    finish='return finish_vectors(engine,l,u,stats)' if 'return finish_vectors' in source else 'return l,u,stats'
    prefix='l,u,stats=finish_vectors(engine,l,u,stats)\n    ' if 'finish_vectors' in finish else ''
    return study.infra.cached_clone(fn,[
        ('d=len(theta)-1;', 'engine.iq_batches=[]\n    d=len(theta)-1;',1),
        (finish,prefix+'return complete_vectors(engine,theta,plan,queue,t,fl,kind,ids,teachers,l,u,stats)',1)],
        dict(complete_vectors=complete_vectors))


def engine_wrapper(parent):
    wrapped=ORIGINAL_ENGINE(parent)
    class Engine(wrapped):
        def gradient(self,theta,ids,k,kind='ce'):
            result=super().gradient(theta,ids,k,kind)
            if kind=='ce' and hasattr(self,'iq_batches'):
                self.iq_batches.append((self.sample_owner[int(ids[0])],list(ids),k))
            return result
        def finish_work(self):
            result=super().finish_work();result['manuscript_curvature']=self.iq_curvature
            return result
    return Engine


def run(variant):
    global ACTIVE,ORIGINAL_VECTORS,ORIGINAL_ENGINE
    protocol=register();ACTIVE=variant
    selection.install();study.register()
    ORIGINAL_VECTORS=study.vectors;ORIGINAL_ENGINE=study.engine_wrapper
    study.vectors=vectors;study.engine_wrapper=engine_wrapper
    old_make=study.make_arm
    def make(base,deployment,radios,cap):
        class Radio:
            @staticmethod
            def analog(l,u,ch,c,seed):return radio.analog(l,u,ch,c,seed,CURRENT_ELL)
        radios[study.PROPOSED]=Radio
        return old_make(base,deployment,radios,cap)
    study.make_arm=make
    study.OUT=OUT/variant;study.register=lambda:protocol
    if variant=='paper_iq':
        study.VARIANTS['paper_iq']=dict(label=VARIANTS[variant],objective='paper',selection=0,equal=True,exact=False)
    study.run('gate_select16' if variant=='iq_only' else 'paper_iq','baseline')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','run']);parser.add_argument('--variant',choices=list(VARIANTS))
    args=parser.parse_args()
    if args.action=='register':print(register())
    else:run(args.variant)
