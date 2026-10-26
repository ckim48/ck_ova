"""Retain-only proxy distillation; isolated, preregistered radio comparison.

All main radio arms share the new objective. Evaluation oracle is never a teacher.
Frozen v2/v3 code and results are read-only. Only one process holds the GPU lock.
"""
from pathlib import Path
import sys, json, time, os, hashlib, fcntl, traceback, math, importlib.util
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT/'tools/overnight_energy_v2/retrieval_fourway_v2'
LONG = ROOT/'tools/overnight_energy_v2/retrieval_long200_v3'
SOURCE = ROOT/'results/overnight_energy_v2/retrieval_long200_v3'
OUT = ROOT/'results/overnight_energy_v2/proxy_objective_v1'
sys.path[:0] = [str(FROZEN), str(ROOT), str(ROOT/'tools/overnight_energy_v2'),
               str(ROOT/'tools/overnight_energy_v2/initialization_v11'),
               str(ROOT/'tools/overnight_energy_v2/receiver_control_v1')]
METHODS = ['proposed', 'aircomp_only', 'aircomp_oma', 'oma_only', 'hybrid_no_crossmodal']
PROXY_ROUNDS = 20
LABELS = dict(proposed='NOMA+AirComp',aircomp_only='AirComp+AirComp (one sum)',
              aircomp_oma='OMA+AirComp',oma_only='OMA-only',
              hybrid_no_crossmodal='NOMA+AirComp without cross-modal preservation')

def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(path)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def status(**x): write(OUT/'status.json',dict(unix=time.time(),pid=os.getpid(),**x))
def lr_at(k): return .0035 if k<=40 else .0035*.5*(1+math.cos(math.pi*(k-40)/160))
def mode(method): return 'hybrid' if method in ['proposed','hybrid_no_crossmodal'] else method

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def proxy_loss(images,texts,old_images,old_texts,scale):
    import torch.nn.functional as F
    logits=scale.exp()*images@texts.T
    old=scale.detach().exp()*old_images.detach()@old_texts.detach().T
    return .5*(F.kl_div(F.log_softmax(logits,1),F.softmax(old,1),reduction='batchmean')+
               F.kl_div(F.log_softmax(logits.T,1),F.softmax(old.T,1),reduction='batchmean'))

def cpu_checks():
    import torch
    import torch.nn.functional as F
    torch.manual_seed(817)
    im=F.normalize(torch.randn(4,12,dtype=torch.float64),dim=-1)
    tx=F.normalize(torch.randn(20,12,dtype=torch.float64),dim=-1)
    scale=torch.tensor(1.,dtype=torch.float64)
    assert abs(float(proxy_loss(im,tx,im,tx,scale)))<1e-12
    variable=(im+.1*torch.randn_like(im)).requires_grad_(True)
    value=proxy_loss(variable,tx,im,tx,scale)
    grad=torch.autograd.grad(value,variable)[0]
    after=proxy_loss(variable-.001*grad,tx,im,tx,scale)
    assert after<value
    duplicated=proxy_loss(variable.detach(),tx.repeat_interleave(2,0),im,tx.repeat_interleave(2,0),scale)
    assert torch.allclose(value.detach(),duplicated,atol=1e-12)
    perm=torch.randperm(len(tx))
    assert torch.allclose(value.detach(),proxy_loss(variable.detach(),tx[perm],im,tx[perm],scale),atol=1e-12)
    return dict(identity_zero=True,gradient_descent_sign=True,duplicate_caption_invariance=True,
                gallery_permutation_invariance=True,bidirectional_all_caption_loss=True)

class Teacher:
    def __init__(self,engine,theta):
        self.engine=engine;self.theta=theta.clone().detach();self.images={};self.texts={}
    def features(self,ids,cids):
        import torch
        import torch.nn.functional as F
        e=self.engine
        with torch.no_grad():
            v,t,_=e.model.parts(self.theta)
            missing=[i for i in ids if i not in self.images]
            if missing:
                z=F.normalize(e.model.image_features(e.data.pixels(missing,e.device),v),dim=-1)
                self.images.update({i:f.cpu() for i,f in zip(missing,z)})
            missing=[i for i in cids if i not in self.texts]
            for j in range(0,len(missing),64):
                batch=missing[j:j+64]
                z=F.normalize(e.model.text_features(**e.data.texts(batch,e.device),text=t),dim=-1)
                self.texts.update({i:f.cpu() for i,f in zip(batch,z)})
        return (torch.stack([self.images[i] for i in ids]).to(e.device),
                torch.stack([self.texts[i] for i in cids]).to(e.device))

def forgetting_gradient(engine,theta,ids,teacher):
    import torch
    if len(ids)<2:return torch.zeros_like(theta),0.
    cids=[c for i in ids for c in engine.data.caption_ids[i]]
    old=teacher.features(ids,cids)
    variable=theta.detach().requires_grad_(True)
    im,tx,scale=engine.features(variable,ids,cids)
    value=proxy_loss(im,tx,*old,scale)
    grad=torch.autograd.grad(value,variable)[0].detach();grad[-1]=0
    assert torch.isfinite(grad).all()
    return grad,float(value.detach())

def new_updates(engine,theta,rec,k,forget,teacher,crossmodal=True):
    lr=engine.plan['learning_rate'];n=len(rec['retained']);d=len(theta)-1
    l=np.zeros((n,d));u=np.zeros_like(l);stats=[]
    count=sum(map(len,rec['retained']));nf=sum(map(len,forget))
    for user,ids in enumerate(rec['batches']):
        g,loss=engine.gradient(theta,ids,k)
        l[user]=(-lr*len(rec['retained'][user])/count*g[:-1]).cpu().numpy()
        row=dict(user=user,retained_loss=loss)
        if forget[user] and k in [21,22,23,24,25]:
            fids=np.random.default_rng(9000+k*10+user).choice(forget[user],min(16,len(forget[user])),replace=False).astype(int).tolist()
            fg,fl=forgetting_gradient(engine,theta,fids,teacher)
            rg,reg=engine.gradient(theta,ids,k,'relation') if crossmodal else (g*0.,0.)
            update=fg+rg
            u[user]=(-lr*len(forget[user])/nf*update[:-1]).cpu().numpy()
            row.update(forget_proxy_KL=fl,retained_relation_KL=reg,
                       forget_gradient_norm=float(fg.norm()),preserve_gradient_norm=float(rg.norm()),
                       forget_images=len(fids),forget_captions=5*len(fids))
        stats.append(row)
    return l,u,stats

def load_theta(path,device):
    import torch
    return torch.load(path,map_location='cpu',weights_only=True)['theta'].to(device)

def save_theta(path,theta,k):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(dict(theta=theta.cpu(),round=k),tmp);tmp.replace(path)

def reference(k):
    import torch
    z=np.load(SOURCE/f'retrain/probabilities_{k:03d}.npz')
    return tuple(torch.from_numpy(z[key].copy()) for key in ['log_image','log_text'])

def evaluate(engine,theta,k,requests):
    import torch
    e,prob=engine.evaluate(theta,reference(200))
    gallery=engine.plan['gallery_images']
    def extra(ids,ref):
        ids=[i for i in ids if i in gallery]
        if not ids:return dict(images=0)
        ii=torch.tensor([gallery.index(i) for i in ids]);ci=(ii[:,None]*5+torch.arange(5)).flatten()
        a,b=ref;li,lt=prob
        return dict(images=len(ids),KL_retrain_to_candidate=max(0.,float(((a[ii].exp()*(a[ii]-li[ii])).sum(1).mean()+
                   (b[ci].exp()*(b[ci]-lt[ci])).sum(1).mean())/2)),
                    i2t_top1_disagreement=float((a[ii].argmax(1)!=li[ii].argmax(1)).float().mean()),
                    t2i_top1_disagreement=float((b[ci].argmax(1)!=lt[ci].argmax(1)).float().mean()))
    e['matched_horizon_retrain']={name:extra(ids,reference(k)) for name,ids in
              [('retained',engine.plan['validation_images']),('forget',engine.plan['forget_probe'])]}
    e['per_request']={str(u):extra(ids,reference(200)) for u,ids in enumerate(requests) if ids}
    return e,prob

def proxy_schedule(records,forgotten):
    result=[]
    for k in range(1,PROXY_ROUNDS+1):
        pools=[sorted(set(p)-forgotten) for p in records[0]['retained']]
        batches=[np.random.default_rng(240000+k*10+u).choice(p,min(16,len(p)),replace=False).astype(int).tolist()
                 for u,p in enumerate(pools)]
        assert all(not (set(p)&forgotten) for p in batches)
        result.append(dict(round=k,retained=pools,batches=batches))
    return result

def train_proxy(engine,records,transport,channels,config,name):
    from train import update_theta
    dest=OUT/'proxies'/name;dest.mkdir(parents=True,exist_ok=True)
    final=dest/'complete.json'
    if final.exists():return load_theta(dest/'round_020.pt',engine.device),json.loads(final.read_text())
    theta=engine.theta0.clone();e=ee=tt=0.;rows=[]
    for rec in records:
        k=rec['round'];t0=time.time();status(stage='proxy_training',proxy=name,round=k,total=PROXY_ROUNDS)
        engine.plan['learning_rate']=.0035
        l,_,_=engine.updates(theta,rec,k,'retrain',[[] for _ in range(10)])
        if name=='ideal':vector=l.sum(0);radio=dict(status='ideal_diagnostic')
        else:
            vector,radio=transport(l,np.zeros_like(l),channels,config,'oma_only' if name=='oma' else 'hybrid',240000+k)
            if vector is None:raise RuntimeError('Proxy failed to find feasible design: '+name)
            if name=='analog' and k==1:
                aliases={}
                for alias in ['aircomp_only','aircomp_oma']:
                    other,meta=transport(l,np.zeros_like(l),channels,config,alias,240000+k)
                    assert other is not None and np.allclose(other,vector,rtol=1e-10,atol=1e-12)
                    assert abs(meta['energy_bound_j']-radio['energy_bound_j'])<1e-10
                    assert abs(meta['uplink_s']-radio['uplink_s'])<1e-12
                    aliases[alias]=dict(max_vector_difference=float(np.max(np.abs(other-vector))),
                                       energy_difference_j=meta['energy_bound_j']-radio['energy_bound_j'])
                write(OUT/'proxy_transport_reuse_check.json',dict(passed=True,aliases=aliases))
            e+=radio['energy_bound_j'];ee+=radio['expected_energy_j'];tt+=radio['uplink_s']
        theta=update_theta(theta,vector)
        row=dict(round=k,radio=radio,wall_s=time.time()-t0);rows.append(row)
        write(dest/f'round_{k:03d}.json',row)
        print(json.dumps(dict(stage='proxy',proxy=name,round=k,wall_s=row['wall_s'])),flush=True)
    save_theta(dest/'round_020.pt',theta,20)
    meta=dict(status='completed',rounds=20,energy_bound_j=e if name!='ideal' else None,
              expected_energy_j=ee if name!='ideal' else None,uplink_s=tt if name!='ideal' else None,
              teacher_sha256=sha(dest/'round_020.pt'),schedule_sha256=sha(OUT/'proxy_records.json'))
    write(final,meta);return theta,meta

def ideal_pilot(engine,records,forget,proxy):
    from train import update_theta
    dest=OUT/'ideal_pilot';dest.mkdir(exist_ok=True)
    if (dest/'gate.json').exists():return json.loads((dest/'gate.json').read_text())
    theta0=load_theta(SOURCE/'proposed/round_020.pt',engine.device)
    engine.set_teacher(theta0);teacher=Teacher(engine,proxy);result={}
    for arm in ['legacy','proxy']:
        theta=theta0.clone()
        for k in range(21,26):
            status(stage='ideal_pilot',arm=arm,round=k);engine.plan['learning_rate']=.0035
            if arm=='legacy':l,u,local=engine.updates(theta,records[k-1],k,'proposed',forget)
            else:l,u,local=new_updates(engine,theta,records[k-1],k,forget,teacher)
            theta=update_theta(theta,(l+u).sum(0))
            write(dest/arm/f'round_{k:03d}.json',dict(round=k,local=local))
        e,_=evaluate(engine,theta,25,forget);result[arm]=e
        save_theta(dest/arm/'round_025.pt',theta,25)
    a,b=result['legacy'],result['proxy']
    checks=dict(final_reference_forget_KL_improved=b['forget']['KL_retrain_to_candidate']<a['forget']['KL_retrain_to_candidate']-1e-6,
      matched_horizon_forget_KL_improved=b['matched_horizon_retrain']['forget']['KL_retrain_to_candidate']<a['matched_horizon_retrain']['forget']['KL_retrain_to_candidate']-1e-6,
      retained_loss_not_degraded=b['retained']['loss']<=a['retained']['loss']+.02,
      retained_mR_not_degraded=b['retained']['mR']>=a['retained']['mR']-.01)
    gate=dict(advance_to_200=all(checks.values()),checks=checks,results=result,
              interpretation='Development gate only; five ideal-link steps, no transport ranking. No settings selected from test.')
    write(dest/'gate.json',gate);print(json.dumps(dict(stage='ideal_gate',**gate)),flush=True)
    return gate

def prefix(method):
    src=SOURCE/('proposed' if method=='hybrid_no_crossmodal' else method)
    rows=json.loads((src/'prefix_rounds.json').read_text())
    rows=[r for r in rows if r['round']<=20]
    assert [r['round'] for r in rows]==list(range(1,21))
    return src,rows

def physical(engine,plan,records,forget,proxies,transport,channels,method,stop):
    from train import update_theta
    dest=OUT/method;dest.mkdir(exist_ok=True);rowsdir=dest/'rounds';rowsdir.mkdir(exist_ok=True)
    manifest=dest/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['status'].startswith('incomplete'):return
    src,old=prefix(method);pk='oma' if method=='oma_only' else 'analog'
    proxy,pcost=proxies[pk]
    teacher=Teacher(engine,proxy)
    prior=sorted(rowsdir.glob('*.json'))
    if prior:
        last=json.loads(prior[-1].read_text());k0=last['round']
        theta=load_theta(dest/f'round_{k0:03d}.pt',engine.device)
        e=last['cumulative_tx_energy_bound_j'];ee=last['cumulative_tx_energy_expected_j'];tt=last['cumulative_uplink_s']
    else:
        k0=20;theta=load_theta(src/'round_020.pt',engine.device)
        last=old[-1];e=last['cumulative_tx_energy_bound_j']+pcost['energy_bound_j']
        ee=last['cumulative_tx_energy_expected_j']+pcost['expected_energy_j'];tt=last['cumulative_uplink_s']+pcost['uplink_s']
        write(dest/'prefix_and_proxy_cost.json',dict(prefix=old[-1],proxy=pcost,prefix_checkpoint_sha256=sha(src/'round_020.pt'),
             all_costs_included=True,proxy_training_charged_per_independent_arm=True))
        save_theta(dest/'round_020.pt',theta,20)
    engine.set_teacher(load_theta(src/'round_020.pt',engine.device))
    for k in range(k0+1,stop+1):
        t0=time.time();status(stage='physical',method=method,round=k,stage_stop=stop,max_rounds=200)
        engine.plan['learning_rate']=lr_at(k)
        l,u,local=new_updates(engine,theta,records[k-1],k,forget,teacher,method!='hybrid_no_crossmodal')
        vector,radio=transport(l,u,channels,plan['config'],mode(method),100000+k)
        if vector is None:
            write(manifest,dict(status='incomplete_no_feasible_design_found',failed_round=k,radio=radio));return
        e+=radio['energy_bound_j'];ee+=radio['expected_energy_j'];tt+=radio['uplink_s']
        theta=update_theta(theta,vector)
        row=dict(method=method,round=k,radio=radio,learning_rate=lr_at(k),local=local,
          cumulative_tx_energy_bound_j=e,cumulative_tx_energy_expected_j=ee,cumulative_uplink_s=tt,
          proxy_training_rounds=PROXY_ROUNDS,request_algorithm_completed=k>=25,logical_requests=5 if k>=21 else 0)
        if k==21 or k%5==0:
            ev,prob=evaluate(engine,theta,k,forget);row['evaluation']=ev
            np.savez_compressed(dest/f'probabilities_{k:03d}.npz',log_image=prob[0].numpy(),log_text=prob[1].numpy())
        save_theta(dest/f'round_{k:03d}.pt',theta,k)
        row['wall_s']=time.time()-t0;write(rowsdir/f'{k:03d}.json',row)
        if 'evaluation' in row:report(plan)
        print(json.dumps(dict(stage='physical',method=method,round=k,wall_s=row['wall_s'],evaluation=row.get('evaluation'))),flush=True)
    mia,_=engine.mia_scores(theta)
    oldmia=json.loads((SOURCE/'retrain/mia.json').read_text())
    mia['auc_gap_vs_retrain']=abs(mia['loss_score_auc']-oldmia['loss_score_auc'])
    write(dest/f'mia_{stop:03d}.json',mia)
    write(manifest,dict(status='completed_200' if stop==200 else 'checkpoint_40',rounds=stop))
    report(plan)

def report(plan):
    ref=json.loads((SOURCE/'retrain/rounds/200.json').read_text())['evaluation']['retained']['mR']
    target=.95*ref;result=[]
    for method in METHODS:
        dest=OUT/method;rows=[json.loads(p.read_text()) for p in sorted((dest/'rounds').glob('*.json'))]
        evs=[r for r in rows if 'evaluation' in r]
        manifest=json.loads((dest/'manifest.json').read_text()) if (dest/'manifest.json').exists() else {'status':'pending_or_running'}
        out=dict(method=method,label=LABELS[method],status=manifest['status'],completed_rounds=rows[-1]['round'] if rows else 20)
        def point(row):
            return dict(round=row['round'],retained_mR=row['evaluation']['retained']['mR'],retained_loss=row['evaluation']['retained']['loss'],
              forget_KL=row['evaluation']['forget']['KL_retrain_to_candidate'],energy_bound_j=row['cumulative_tx_energy_bound_j'],
              energy_expected_j=row['cumulative_tx_energy_expected_j'],uplink_s=row['cumulative_uplink_s'])
        if evs:out['last']=point(evs[-1])
        hits=[]
        for row in evs:
            e=row['evaluation'];passed=row['round']>=25 and e['retained']['mR']>=target and e['forget']['KL_retrain_to_candidate']<=.01
            hits=hits+[row] if passed else []
            if len(hits)>=3:out['joint_target']=dict(entry_round=hits[0]['round'],confirmation=point(row));break
        out.setdefault('joint_target',None)
        for metric,values in [('mR',[.25,.29,.30]),('loss',[4.1,4.,3.9])]:
            targets={}
            for value in values:
                found=next((r for r in evs if r['round']>=25 and
                   (r['evaluation']['retained']['mR']>=value if metric=='mR' else r['evaluation']['retained']['loss']<=value)),None)
                targets[str(value)]=point(found) if found else None
            out['utility_only_'+metric+'_targets']=targets
        mias=sorted(dest.glob('mia_*.json'))
        if mias:out['mia']=json.loads(mias[-1].read_text())
        result.append(out)
    write(OUT/'summary.json',dict(methods=result,retained_mR_threshold=target,forget_KL_max=.01,
      confirmation='3 consecutive scheduled evaluations after request completion; costs at confirmation include proxy construction',
      main_comparison='All four main arms share proxy and cross-modal objective; no-crossmodal is a separate paired ablation.',
      scope='Single seed0; no test access; same old v3 channels/codec/power/deadline. TX waveform upper bound, expected energy and uplink occupied time only.',
      ranking_policy='Failed, pending, quality-missed and ideal-reference trajectories are not energy-to-target winners.'))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'cpu_checks.json',cpu_checks())
    if '--cpu-only' in sys.argv:print('CPU_CHECKS_PASSED');return
    lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(),'Refuse duplicate run; explicit recovery required.'
    plan=json.loads((SOURCE/'plan.json').read_text())
    for n,h in plan['sources'].items():assert sha(ROOT/n)==h,n
    for n,h in plan['frozen_engine_hashes'].items():assert sha(FROZEN/n)==h,n
    for n,h in plan['runner_hashes'].items():assert sha(LONG/n)==h,n
    import torch
    from train import Engine
    from dataset import RetrievalData
    transport=load_module('proxy_frozen_partial_transport',LONG/'radio_partial.py').communicate
    records=json.loads((SOURCE/'records.json').read_text());forget=records[20]['forget']
    schedules=proxy_schedule(records,set(plan['forgotten_images']))
    write(OUT/'proxy_records.json',schedules)
    protocol=dict(version='proxy_objective_v1',seed=0,methods=METHODS,main_methods=METHODS[:4],
      proposal='ConMU-inspired VLM extension, not exact ConMU reproduction. Replace negative forget CE by bidirectional KL(proxy||candidate), preserve retained relation KL.',
      weights=dict(forget_proxy=1.,retain_relation=1.),proxy_rounds=PROXY_ROUNDS,
      proxy='Fresh frozen CLIP/prompt initialization, retained training data only, independent seed240000 minibatches. No validation/test gallery, evaluation-reference checkpoint or predictions used for training.',
      forgetting='All5 captions per sampled forget image, local gallery16 images/80captions. Same five request steps21..25 and SGD learning rates across methods.',
      preservation='Current pre-request-model retained relation KL, separate from proxy teacher; baseline main arms receive identical term.',
      horizons='Ideal pilot21..25; all five physical arms through40 regardless of gate. If ideal proxy reduces both reference KLs without retained loss increase>.02 or mR drop>.01, extend every feasible arm to200.',
      proxy_cost='Each independent arm pays full20-round federated proxy construction. Analog-learning proxy reused exactly for three modes with identical no-payload transport; OMA-only builds its own digital proxy. Ideal diagnostic proxy is never an energy-ranked arm.',
      physical='Frozen partial-overlap v3 with same Q8 canonical deletion target, perfect SIC/CSI; no receiver-side nonlinear update controller.',
      prefix='Existing matching rounds1..20 checkpoints and all costs retained separately by transport.',
      evaluation='Existing final200 and matched-horizon retrain are independent reporting/development references, never teachers. All coefficients/budgets fixed before new outcomes; no test access.',
      targets=plan['quality_thresholds'],frozen_plan_sha256=sha(SOURCE/'plan.json'),
      code_sha256=sha(__file__),frozen_engine_hashes=plan['frozen_engine_hashes'])
    write(OUT/'protocol.json',protocol);write(OUT/'started.json',dict(pid=os.getpid(),unix=time.time()))
    status(stage='loading_model');engine=Engine(RetrievalData(),dict(plan));channels=np.load(SOURCE/'channels.npy')
    ideal,_=train_proxy(engine,schedules,transport,channels,plan['config'],'ideal')
    gate=ideal_pilot(engine,records,forget,ideal)
    proxies={name:train_proxy(engine,schedules,transport,channels,plan['config'],name) for name in ['analog','oma']}
    for method in METHODS:physical(engine,plan,records,forget,proxies,transport,channels,method,40)
    write(OUT/'first40_complete.json',dict(unix=time.time(),gate=gate['advance_to_200']))
    if gate['advance_to_200']:
        for method in METHODS:physical(engine,plan,records,forget,proxies,transport,channels,method,200)
    for n,h in plan['frozen_engine_hashes'].items():assert sha(FROZEN/n)==h,n
    write(OUT/'finished.json',dict(unix=time.time(),extended_to_200=gate['advance_to_200']))
    status(stage='completed',extended_to_200=gate['advance_to_200']);report(plan)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status(stage='failed',error=repr(exc));traceback.print_exc();raise
