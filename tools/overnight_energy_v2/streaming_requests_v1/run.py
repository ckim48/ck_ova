"""Repeated causal FU requests with parallel or periodic serial uplink service.

Preserves previous experiments. Six trajectories: four requested main schemes
and two immediate serial controls. No outcomes select a scheduling period.
"""
from pathlib import Path
import os,sys,time,json,hashlib,fcntl,subprocess,importlib.util,traceback
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/streaming_requests_v1'
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/parallel_aircomp_v1'))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('stream_parent',ROOT/'tools/overnight_energy_v2/parallel_aircomp_v1/run.py')
mod=load('stream_modality',ROOT/'tools/overnight_energy_v2/parallel_modality_v2/run.py')
core=base.core
from scheduler import Queue,make_requests,self_check,replay,FRAME_S,HORIZON,FL_STEPS,STEPS
import radio_stream as radio
ARMS={
 'parallel_crossmodal':dict(parallel=True,crossmodal=True,period=0,digital=False,main=True),
 'separate_aircomp_periodic':dict(parallel=False,crossmodal=True,period=5,digital=False,main=True),
 'parallel_modalitywise':dict(parallel=True,crossmodal=False,period=0,digital=False,main=True),
 'aircomp_ofdma_periodic':dict(parallel=False,crossmodal=True,period=5,digital=True,main=True),
 'separate_aircomp_immediate':dict(parallel=False,crossmodal=True,period=0,digital=False,main=False),
 'aircomp_ofdma_immediate':dict(parallel=False,crossmodal=True,period=0,digital=True,main=False)}

def write(name,obj):core.write(OUT/name,obj)
def status(**obj):write('status.json',dict(unix=time.time(),pid=os.getpid(),**obj))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);plan,parent=base.prepare()
    records=json.loads((core.SOURCE/'records.json').read_text());cohort=records[20]['forget']
    seen=[set() for _ in range(10)]
    for rec in records[:20]:
        for u,ids in enumerate(rec['batches']):seen[u].update(ids)
    assert all(set(ids)<=seen[u] for u,ids in enumerate(cohort))
    requests=make_requests(cohort)
    assert set(i for r in requests for i in r['images'])==set(plan['forgotten_images'])
    assert set(plan['forgotten_images']).isdisjoint(plan['validation_images']+plan['official_test_images'])
    # A fixed proxy on a declared non-revocable subset is available before requests.
    # It uses no request arrival times or future model/reference predictions.
    pm=ROOT/'results/overnight_energy_v2/proxy_objective_v2/proxies/analog/complete.json'
    proxy_meta=json.loads(pm.read_text())
    ps=[ROOT/f'results/overnight_energy_v2/{v}/proxy_records.json' for v in ['proxy_objective_v2','proxy_objective_v1']]
    schedule=next(p for p in ps if p.exists() and core.sha(p)==proxy_meta['schedule_sha256'])
    safe=json.loads(schedule.read_text());revocable=set(plan['forgotten_images'])
    assert all(revocable.isdisjoint(ids) for rec in safe for ids in rec['batches'])
    files=[*HERE.glob('*.py'),ROOT/'tools/overnight_energy_v2/parallel_modality_v2/run.py',
           pm,schedule,core.SOURCE/'proposed/round_020.pt',core.SOURCE/'retrain/round_200.pt']
    hashes=dict(parent['source_hashes']);hashes.update({str(p.relative_to(ROOT)):core.sha(p) for p in files})
    p=OUT/'protocol.json'
    spec=dict(version='streaming_requests_v1',registered_unix=time.time(),arms=ARMS,seed=0,
      authorization='User explicitly required repeated need-driven FU, joint immediate service for parallel AirComp, and periodic FU with paused FL uplink for separated AirComp/OFDMA.',
      initial_model='Reuse common completed20-round model. All558 revocable images have already appeared in common warmup training batches.',
      arrivals='Nine bursts, five UAV requests per burst, disjoint partitions of the same558 previously learned images. Seed231023 cohort shuffling and seed9123 jitter; every method sees exactly the same exogenous arrival times.',
      causal_retention='Remove requested images from subsequent retained FL batches immediately upon arrival, even if FU service is queued. A transmission already started is not retroactively changed; later FU must handle its effect.',
      clock='Uplink service clock only: one normalized frame=20480/1MHz=20.48ms. No local-compute, downlink, control or decoding delay is modeled. Thus request elapsed time is a communication-service lower bound, not end-to-end user latency or a claimed real request arrival rate.',
      workload='Bursts spaced nominally20 normalized frames with fixed +-3-frame jitter. This is an explicitly frequent-request stress workload. CPU timing-only sensitivity also reports nominal40/60-frame intervals.',
      scheduling='Parallel sends FL plus all active FU in one analog aggregate. Separated schemes batch at fixed5-frame boundaries and perform five FU-only updates, pausing FL uploads during these phases. New arrivals wait for the next batch. FL can use the channel outside FU phases; no assumption that local FL computation is impossible.',
      stronger_controls='Two additional serial arms serve pending FU immediately, avoiding arbitrary periodic waiting. CPU timing-only sensitivity covers immediate and1/5/10/20-frame periods; all are reported, none selected from quality outcomes.',
      training='All arms complete180 further FL updates (200 incl. warmup) and all45 requests unless physically infeasible. Identical SGD FL cosine schedule indexed by completed FL update count. Each request gets5 FU steps at original initial FU rate0.0035, so late requests are not disabled by the terminal FL learning rate.',
      retained_proxy='Reuse fixed safe-subset proxy20, excluding the entire predeclared revocable cohort from its construction. This requires a known non-revocable training subset and is NOT retraining on the current retained set after each arrival. The same assumption, teacher and construction costs apply to every arm. Future request times do not enter proxy construction.',
      FU='Same bidirectional image-text proxy KL plus retained relation KL as before; retained teacher is snapshotted when each request first receives service. Modality-wise arm replaces both with intra-modal KL. Request-local batches/seeds and weights are otherwise identical.',
      reference='Existing final200 retrain checkpoint excludes exactly the same union558 images. It is reporting-only, never a teacher. Re-evaluate it on the new gallery256validation+558forget images. Early distances to this final-set reference are descriptive, not a current-request deletion certificate. Quality targets apply after all requests arrive and complete service.',
      quality='Retained validation mR >=95% of final retained retrain on the SAME new gallery; mean and worst request KL<=0.01, all45 requests have5 steps. Report loss, both retrieval directions, prediction disagreement and MIA separately. Procedure completion and quality success are distinct.',
      deadline='Each request has a common20-frame communication-service deadline. Late requests are still served and reported. Meeting five-step deadlines is not proof of successful deletion.',
      radio='Same10 heads,1MHz,0.32W user power,0.5J per-user per-transaction limit,0.05J network TX per-transaction cap,128 coordinates/block and actual128-tone OFDMA Q6. Pure FU digital has no artificial FL slot. Analog has no quantization. Continuous noise/energy optimization within common exact-inversion family.',
      error_budget='For joint FL+FU: epsilon^2=(0.2*FL blocknorm bound)^2+(0.2*FU blocknorm bound)^2. Separate components each receive their corresponding bound; quantization included for digital FU. This new rule supports FU-only transactions and is explicitly different from old FL-only normalization.',
      evaluation_axes='Record FL updates and FU completions at same180-frame adaptation horizon, and costs/delay for the same200 FL updates and request service. Report occupied uplink and elapsed service-clock time (includes queue/idle waiting) separately, plus request latency/backlog/deadline misses.',
      costs='Every arm pays the same initial20-round model and safe-proxy20 construction costs. During adaptation charge all actual transmissions including FU-only phases. Ideal retrain is not an energy-ranked scheme.',
      limitations='Single-seed versioned system experiment, not a published scheme reproduction, not proof of novelty. Periodic FU is a scheduling policy, not an inherent OFDMA requirement. Fixed safe-subset teacher and perfect-CSI/ideal coding assumptions must be disclosed.',
      source_hashes=hashes)
    if p.exists():spec=json.loads(p.read_text())
    else:write('protocol.json',spec);write('requests.json',requests)
    assert json.loads((OUT/'requests.json').read_text())==requests
    for name,h in spec['source_hashes'].items():assert core.sha(ROOT/name)==h,name
    plan=dict(plan,gallery_images=sorted(set(plan['validation_images']+plan['forgotten_images'])),
              forget_probe=plan['forgotten_images'])
    return plan,requests

def local_gradient(engine,theta,ids,cids,teacher,crossmodal):
    import torch
    variable=theta.detach().requires_grad_(True)
    im,tx,scale=engine.features(variable,ids,cids);old=teacher.features(ids,cids)
    value=(core.proxy_loss if crossmodal else mod.intra_loss)(im,tx,*old,scale)
    g=torch.autograd.grad(value,variable)[0].detach();g[-1]=0
    assert torch.isfinite(g).all()
    return g,float(value.detach())

def retained_pools(plan,queue,t):
    deleted=set(i for r in queue.arrived(t) for i in r['images'])
    pools=[sorted(set(p)-deleted) for p in plan['partitions']]
    assert all(len(p)>=2 for p in pools)
    return pools

def vectors(engine,theta,plan,queue,t,fl,kind,ids,proxy,teachers,crossmodal):
    d=len(theta)-1;l=np.zeros((10,d));u=np.zeros_like(l);stats=[]
    pools=retained_pools(plan,queue,t)
    if kind in ['fl','joint'] and fl<FL_STEPS:
        k=21+fl;lr=core.lr_at(k);total=sum(map(len,pools))
        for user,pool in enumerate(pools):
            batch=np.random.default_rng(730000+k*10+user).choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
            g,_=engine.gradient(theta,batch,k)
            l[user]=(-lr*len(pool)/total*g[:-1]).cpu().numpy()
    total=sum(len(queue.items[i]['images']) for i in ids)
    for rid in ids:
        req=queue.items[rid];user=req['user'];step=req['steps'];fids=req['images']
        if rid not in teachers:teachers[rid]=core.Teacher(engine,theta)
        cids=[c for i in fids for c in engine.data.caption_ids[i]]
        fg,loss=local_gradient(engine,theta,fids,cids,proxy,crossmodal)
        seed=910000+req['burst']*1000+user*10+step
        rids=np.random.default_rng(seed).choice(pools[user],min(16,len(pools[user])),replace=False).astype(int).tolist()
        rcids=engine.data.training_caption_ids(rids,21+step)
        rg,reg=local_gradient(engine,theta,rids,rcids,teachers[rid],crossmodal)
        u[user]+=(-.0035*len(fids)/total*(fg+rg)[:-1]).cpu().numpy()
        stats.append(dict(request=rid,step=step+1,forget_loss=loss,preservation_loss=reg,
                          forget_gradient_norm=float(fg.norm()),preserve_gradient_norm=float(rg.norm()),
                          images=len(fids),retained_images=rids))
    return l,u,stats

def evaluation(engine,theta,ref_probs,queue,t):
    import torch
    e,prob=engine.evaluate(theta,ref_probs);li,lt=prob;a,b=ref_probs
    gallery=engine.plan['gallery_images'];index={im:i for i,im in enumerate(gallery)}
    per={}
    for req in queue.arrived(t):
        ii=torch.tensor([index[i] for i in req['images']]);ci=(ii[:,None]*5+torch.arange(5)).flatten()
        per[req['id']]=dict(images=len(ii),steps=req['steps'],completed_s=req['completed_s'],
            KL=max(0.,float(((a[ii].exp()*(a[ii]-li[ii])).sum(1).mean()+(b[ci].exp()*(b[ci]-lt[ci])).sum(1).mean())/2)),
            i2t_disagreement=float((a[ii].argmax(1)!=li[ii].argmax(1)).float().mean()),
            t2i_disagreement=float((b[ci].argmax(1)!=lt[ci].argmax(1)).float().mean()))
    e['per_request']=per;e['worst_request_KL']=max((x['KL'] for x in per.values()),default=None)
    e['reference_scope']='Final retained-set retrain; early distances are descriptive only.'
    return e

def report():
    rows=[]
    for name,opt in ARMS.items():
        p=OUT/name
        item=dict(method=name,main=opt['main'],policy=opt,status='pending')
        for n in ['manifest','latest','common_horizon','mia']:
            f=p/(n+'.json')
            if f.exists():item[n]=json.loads(f.read_text())
        if 'manifest' in item:item['status']=item['manifest']['status']
        elif 'latest' in item:item['status']='running'
        rows.append(item)
    write('summary.json',dict(methods=rows,
          scope='Compare same-time horizons or completed same workload; no inference of quality from procedure completion. Old single-burst results are not reused as streaming results.'))

def run_arm(engine,plan,requests,proxy_theta,ref_probs,threshold,channels,name,opt,prefix_cost):
    from train import update_theta
    import torch
    dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'started.json').exists(),'Duplicate arm'
    core.write(dest/'started.json',dict(unix=time.time(),policy=opt))
    theta=core.load_theta(core.SOURCE/'proposed/round_020.pt',engine.device)
    q=Queue(requests,opt['parallel'],opt['period']);teachers={};proxy=core.Teacher(engine,proxy_theta)
    t=occupied=energy=0.;fl=seq=0;horizon_saved=False;target=None;hits=[]
    last_eval=None;last_eval_fl=-1
    def point(at,ev=None):
        return dict(sequence=seq,completed_FL_rounds=20+fl,adaptation_elapsed_service_s=at,
            adaptation_uplink_occupied_s=occupied,adaptation_tx_energy_bound_j=energy,
            cumulative_uplink_occupied_s=prefix_cost['uplink_s']+occupied,
            cumulative_elapsed_service_s=prefix_cost['uplink_s']+at,
            cumulative_tx_energy_bound_j=prefix_cost['energy_j']+energy,
            requests=q.metrics(at),evaluation=ev,joint_target=target)
    for _ in range(2000):
        kind,ids=q.decision(t,fl)
        if kind=='done':break
        if kind=='wait':
            if not horizon_saved and t<=HORIZON<ids:
                ev=evaluation(engine,theta,ref_probs,q,HORIZON)
                core.write(dest/'common_horizon.json',point(HORIZON,ev));horizon_saved=True
            t=ids;continue
        status(stage='training',method=name,transaction=seq+1,FL_round=20+fl,clock_s=t,requests=q.metrics(t))
        l,u,local=vectors(engine,theta,plan,q,t,fl,kind,ids,proxy,teachers,opt['crossmodal'])
        seed=500000+seq
        vector,rm=(radio.digital_fu(u,channels,plan['config']) if kind=='fu' and opt['digital']
                   else radio.analog(l,u,channels,plan['config'],seed))
        if vector is None:
            core.write(dest/'manifest.json',dict(status='incomplete_no_feasible_radio_design',kind=kind,radio=rm,**point(t)))
            report();return
        assert rm['conditional_rmse']<=rm['epsilon']*(1+2e-5)+1e-10
        start=t;end=t+rm['uplink_s']
        # Model update becomes visible only after the whole transmission completes.
        if not horizon_saved and start<=HORIZON<end-1e-10:
            ev=evaluation(engine,theta,ref_probs,q,HORIZON)
            snap=point(HORIZON,ev)
            snap['inflight']=dict(kind=kind,start_s=start,end_s=end,
                 note='Model update not yet applied. Airtime is exact; energy is an interval because within-transaction power varies by block. Do not rank a point estimate as measured accrued energy.')
            snap['adaptation_uplink_occupied_s']+=HORIZON-start
            snap['cumulative_uplink_occupied_s']+=HORIZON-start
            snap['adaptation_energy_interval_j']=[energy,energy+rm['energy_bound_j']]
            snap['cumulative_energy_interval_j']=[prefix_cost['energy_j']+energy,prefix_cost['energy_j']+energy+rm['energy_bound_j']]
            snap['adaptation_tx_energy_bound_j']=None
            snap['cumulative_tx_energy_bound_j']=None
            core.write(dest/'common_horizon.json',snap);horizon_saved=True
        theta=update_theta(theta,vector)
        if kind in ['fl','joint'] and fl<FL_STEPS:fl+=1
        q.commit(ids,start,end);t=end;occupied+=rm['uplink_s'];energy+=rm['energy_bound_j'];seq+=1
        completed_now=any(q.items[i]['completed_s']==end for i in ids)
        due=(fl!=last_eval_fl and fl%5==0) or completed_now or (not horizon_saved and t>=HORIZON-1e-10)
        if due:
            last_eval=evaluation(engine,theta,ref_probs,q,t);last_eval_fl=fl
            done=q.metrics(t)['procedure_completed']==len(requests)
            ok=done and last_eval['retained']['mR']>=threshold and last_eval['forget']['KL_retrain_to_candidate']<=.01 and last_eval['worst_request_KL']<=.01
            hits=hits+[dict(fl_round=20+fl,time_s=t)] if ok else []
            if len(hits)>=3 and target is None:target=dict(confirmation_FL_round=20+fl,elapsed_service_s=prefix_cost['uplink_s']+t,uplink_s=prefix_cost['uplink_s']+occupied,energy_j=prefix_cost['energy_j']+energy,first_pass=hits[0])
            core.write(dest/f'evaluations/{seq:04d}.json',point(t,last_eval))
            core.save_theta(dest/f'checkpoints/{seq:04d}.pt',theta,20+fl)
            if not horizon_saved and t>=HORIZON-1e-10:
                core.write(dest/'common_horizon.json',point(HORIZON,last_eval));horizon_saved=True
        row=dict(**point(t,last_eval if due else None),kind=kind,FU_request_ids=ids,radio=rm,local=local)
        core.write(dest/f'transactions/{seq:04d}.json',row);core.write(dest/'latest.json',row)
        if due:report()
        if t>2000*FRAME_S:raise RuntimeError('Unexpected service clock >2000 frames')
    else:raise RuntimeError('Transaction safety limit')
    final=evaluation(engine,theta,ref_probs,q,t)
    if not horizon_saved:core.write(dest/'common_horizon.json',dict(**point(HORIZON,final),note='Workload finished before common horizon; no extra training.'))
    mia,_=engine.mia_scores(theta);refmia=json.loads((OUT/'reference_mia.json').read_text())
    mia['auc_gap_vs_retrain']=abs(mia['loss_score_auc']-refmia['loss_score_auc'])
    core.write(dest/'mia.json',mia);core.save_theta(dest/'final.pt',theta,20+fl)
    core.write(dest/'request_records.json',list(q.items.values()))
    core.write(dest/'manifest.json',dict(status='completed_200FL_and_requests',**point(t,final)))
    report()

def main():
    plan,requests=prepare();write('cpu_checks.json',self_check())
    if '--prepare' in sys.argv:print('PREPARED');return
    own=(OUT/'launch.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(),'Duplicate campaign'
    gpu=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    write('started.json',dict(unix=time.time(),identity=base.identity(os.getpid())))
    status(stage='loading_model')
    from train import Engine
    from dataset import RetrievalData
    engine=Engine(RetrievalData(),dict(plan));channels=np.load(core.SOURCE/'channels.npy')
    p=ROOT/'results/overnight_energy_v2/proxy_objective_v2/proxies/analog'
    meta=json.loads((p/'complete.json').read_text());proxy_theta=core.load_theta(p/'round_020.pt',engine.device)
    pref=json.loads((core.SOURCE/'proposed/prefix_rounds.json').read_text())[19]
    prefix_cost=dict(energy_j=pref['cumulative_tx_energy_bound_j']+meta['energy_bound_j'],uplink_s=pref['cumulative_uplink_s']+meta['uplink_s'])
    write('common_prefix_cost.json',prefix_cost)
    theta=core.load_theta(core.SOURCE/'proposed/round_020.pt',engine.device)
    q=Queue(requests,True,0);ids=[r['id'] for r in q.pending(0)]
    status(stage='fixed_initial_update_preflight')
    l,u,stats=vectors(engine,theta,plan,q,0,0,'joint',ids,core.Teacher(engine,proxy_theta),{},True)
    test={}
    for name,call in [('joint',lambda:radio.analog(l,u,channels,plan['config'],500000)),
                      ('fl',lambda:radio.analog(l,u*0,channels,plan['config'],500000)),
                      ('fu_aircomp',lambda:radio.analog(l*0,u,channels,plan['config'],500000)),
                      ('fu_ofdma',lambda:radio.digital_fu(u,channels,plan['config']))]:
        v,r=call();test[name]=r;assert v is not None,(name,r)
        assert r['conditional_rmse']<=r['epsilon']*(1+2e-5)+1e-10
    write('preflight.json',dict(passed=True,radio=test,local=stats,scope='Fixed actual full20480-coordinate round20 updates; training data only, no performance selection.'))
    sensitivity=[]
    cohort=json.loads((core.SOURCE/'records.json').read_text())[20]['forget']
    for interval in [20,40,60]:
        req=make_requests(cohort,interval)
        sensitivity.append(dict(interval_frames=interval,method='parallel',period_frames=0,
            result=replay(req,True,0,dict(fl=test['fl']['uplink_s'],joint=test['joint']['uplink_s'],fu=test['fu_aircomp']['uplink_s']))))
        for access in ['aircomp','ofdma']:
            for period in [0,1,5,10,20]:
                sensitivity.append(dict(interval_frames=interval,method='separate_'+access,period_frames=period,
                    result=replay(req,False,period,dict(fl=test['fl']['uplink_s'],joint=test['joint']['uplink_s'],fu=test['fu_'+access]['uplink_s']))))
    write('fixed_update_timing_sensitivity.json',dict(rows=sensitivity,
        limitation='CPU scheduling replay uses constant durations from ONE actual update for every transaction. It is NOT a learning/quality experiment or an adaptive channel/payload simulation.'))
    status(stage='reporting_reference_setup')
    reference_theta=core.load_theta(core.SOURCE/'retrain/round_200.pt',engine.device)
    ref_eval,ref_probs=engine.evaluate(reference_theta,None);ref_mia,_=engine.mia_scores(reference_theta)
    threshold=.95*ref_eval['retained']['mR'];write('reference_metrics.json',dict(metrics=ref_eval,retained_mR_threshold=threshold,reporting_only=True))
    write('reference_mia.json',ref_mia)
    for name,opt in ARMS.items():run_arm(engine,plan,requests,proxy_theta,ref_probs,threshold,channels,name,opt,prefix_cost)
    prepare();report();write('finished.json',dict(unix=time.time(),status='all_registered_arms_finished_or_reported_incomplete'));status(stage='completed')

if __name__=='__main__':
    try:main()
    except Exception as e:OUT.mkdir(parents=True,exist_ok=True);status(stage='failed',error=repr(e));traceback.print_exc();raise
    finally:radio.base.close()
