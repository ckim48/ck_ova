"""Paper-core transfers to the common 10-UAV VLM scenario (not full reproductions).

ISCC-P2: equal-client FedSGD and latency-minimal FDMA at fixed batch/locations.
AFL-OTA: ready-device, stale-gradient Alg.1 with normalized channel-inversion OTA.
Both receive the SAME sequential GA->FT deletion extension, absent in the papers.
"""
import argparse
import copy
import collections
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import literature_baselines as lib
import paper_uav_radio as radio
from adaptive_fu import ROOT, read, write, sha

OUT=ROOT/'results/paper_uav_benchmarks_v1_20261028'
METHODS={'iscc_p2_gaft':'ISCC-P2','uav_afl_ota_gaft':'AFL-OTA'}
DATASETS=lib.DATASETS
AXES=copy.deepcopy(lib.AXES)
for a in AXES.values():a['kind']='training'
BATCH=16
BASE=None
PARAMETERS=None
DEPLOYMENT=None
CURRENT_CASE=None


def cases():
    return [dict(id='baseline',axis=None,value=None)]+[dict(id=lib.case_id(a,v),axis=a,value=v) for a,p in AXES.items() for v in p['values'] if v!=p['base']]


def state(dest=None,**kw):write((Path(dest) if dest else OUT)/'status.json',dict(unix=time.time(),pid=os.getpid(),**kw))


def register():
    path=OUT/'protocol.json'
    if path.exists():
        p=read(path)
        for k,v in p['source_sha256'].items():assert sha(ROOT/k)==v,k
        return p
    source=dict(lib.adaptive.register()['source_sha256'])
    for p in [Path(__file__),Path(radio.__file__),Path(lib.__file__)]:source[str(p.relative_to(ROOT))]=sha(p)
    p=dict(registered_unix=time.time(),methods=METHODS,datasets=DATASETS,axes=AXES,cases=cases(),sampling_seed=0,new_training_arms=84,
      user_approved_transfer='Keep current10-UAV/data/routes, transfer core algorithms and explicitly state adaptations.',
      ISCC_P2=dict(citation='Tang et al., IEEE TWC24(4):2647-2662,2025',doi='https://doi.org/10.1109/TWC.2024.3523381',
        paper='https://arxiv.org/abs/2306.02990',code='https://github.com/TheaSherlock/ISCC-UAV',
        preserved='Equal-client single-step aggregation Eq.(4), full digital FDMA Eq.(8), exact fixed-batch/position P2 Eq.(24) via bisection. Transmissions begin when each local compute finishes.',
        changed='Cached aerial video-text retrieval replaces radar human motion classification. q_s=1 because common cached data already collected; no artificial sensor failures. Same externally prescribed batch caps and200FL. Fixed nominal site head replaces ground server. Full32-bit20480-coordinate adapter upload plus64metadata bits.',
        omitted='Not full BBPO: no P3 batch optimization using unknown VLM convex-loss constants, no P4 sensing/position optimization, no optimal-round search. Named ISCC-P2, not BBPO.'),
      AFL_OTA=dict(citation='Zhai et al., IEEE TWC24(3):2358-2372,2025',doi='https://doi.org/10.1109/TWC.2024.3520501',paper='https://arxiv.org/abs/2403.06653',
        preserved='Alg.1 cached gradients from last received model, all currently ready devices selected, equal-client average Eq.(8)-(10), mean/variance normalization and complex packing Eq.(11)-(16), computation overlaps communication; selected devices receive global model and start next local gradient.',
        changed='Existing UAVs act as clients and nominal site head as server. All-ready selection and unbiased channel inversion with analytic coherent repetition replace Alg.2 joint trajectory/device/transceiver optimizer. No invented fixed staleness delay or restricted participation quota; actual modeled compute availability determines selection. All-peer broadcast circuit accounting is kept for consistency.',
        omitted='Not reproduction of paper joint optimizer or its convergence guarantee. No tuned staleness bound. Error-free mean/variance metadata assumption from paper retained, with zero modeled overhead; listed limitation.'),
      common_FU=dict(source='https://github.com/CSIPlab/SLUG/blob/main/src/clip/unlearn/GAFT.py',reference='https://proceedings.mlr.press/v267/cai25f.html',
        rule='Per request cycle: descend positive image-text cosine, then retained symmetric contrastive CE from updated local model; communicate net two-step delta. Same common SGD cosine LR, retain batch16 and native captions. No clean proxy/request teacher/adaptive controller/Proposed guard.',
        adaptation='Paper baselines are FL-only; GAFT extension is ours and shared across both. Two optimizer substeps per FU cycle, both charged. Not original full-model AdamW implementation.',
        cache_safety='On any new deletion arrival, discard all cached FL updates and drain uninterruptible computations. Every FU uses current model, broadcasts to all, and invalidates stale caches. Never reapply a pre-deletion cache.'),
      fairness='Same feature cache, warm20, requests,200FL,45requests and prescribed FU cycles, data partitions, route, channels, compute/propulsion constants. ISCC/AFL equal-device weighting follows papers; Proposed sample weighting retained. All outcomes kept, single development seed. Learned objectives differ, so not isolated communication comparison. Prior clean proxy gives Proposed additional planned-cohort knowledge; disclose.',
      limits='Fixed32-bit payload; Pmax hard. For paper protocols, network energy/update-energy/round-deadline are reported QoS limits, not grounds to erase a completed workload. Record all exceedances. User previously authorized continuing beyond caps. No physical power relaxation. No hardware-calibrated resource-solver time.',
      accounting='Active FL/FU communication+executed compute+hover, excluding travel/collection/preflight. FL cached jobs billed when created, even if discarded. Pipeline critical path uses actual ready times. Common9TX/1RX uplink circuit accounting and9-peer broadcasts. Shared frozen-encoder cache; adapter-only in-mission costs, not full VLM training.',
      publication='Code, charts, plot arrays and aggregate summaries only. Raw samples/checkpoints/logs local.',source_sha256=source)
    write(path,p);write(OUT/'unit_checks.json',radio.unit());return p


def communication(m):
    import event_energy
    p=event_energy.energy.PARAMETERS['communication'];up=m['uplink_s']
    energy=m['energy_bound_j']/p['pa_efficiency']+up*(9*p['tx_circuit_w']+p['rx_circuit_w'])
    down=0.
    if m.get('broadcast_required',True):
        bits=p['update_coordinates']*p['broadcast_bits_per_coordinate']+p['control_bits_per_update']
        down=bits/p['downlink_rate_bps']
        energy+=down*(p['downlink_power_w']/p['pa_efficiency']+p['tx_circuit_w']+9*p['rx_circuit_w'])
    return energy,down


def train_arm(engine,plan,requests,geo,channels,name,opt,prefix,warm,proxy_theta,refprob,threshold,dest,seed):
    import torch
    from train import update_theta
    from mission_clock import MissionClock,queue_classes
    import event_energy
    base=BASE;physical=event_energy.energy;_,Serial=queue_classes(base)
    ad=dest/name;ad.mkdir(parents=True,exist_ok=True);assert not (ad/'started.json').exists()
    write(ad/'started.json',dict(unix=time.time(),policy=opt))
    q=Serial(requests,False,5);clock=MissionClock(geo,DEPLOYMENT)
    theta=warm.clone();fl=seq=0;t=occupied=rf=0.;last_fl=-1;target=None;hits=[]
    cached={};seen=set();job_number=0;all_jobs=[];current_cost=np.zeros(10)
    async_mode=name=='uav_afl_ota_gaft'
    lib.ACTIVE='fedavg_clip_gaft'
    fu_vectors=lib.vectors(base,'cliperase')
    def point(at,ev=None):
        return dict(sequence=seq,completed_FL_rounds=20+fl,mission_clock_s=at,
            adaptation_elapsed_service_s=at,adaptation_uplink_occupied_s=occupied,adaptation_tx_energy_bound_j=rf,
            requests=q.metrics(at),evaluation=ev,joint_target=target)
    def new_jobs(users,at):
        nonlocal job_number,current_cost
        pools=base.clip.stream.retained_pools(plan,q,at);k=21+fl
        engine.start_work();records=[]
        for user in users:
            pool=pools[user]
            batch=np.random.default_rng(730000+k*10+user).choice(pool,min(BATCH,len(pool)),replace=False).astype(int).tolist()
            g,_=engine.gradient(theta,batch,k)
            records.append((user,g[:-1].detach().cpu().numpy().copy(),batch))
        compute=engine.finish_work();macs=np.array(compute['device_macs']);current_cost+=macs
        _,times=physical.compute_cost(macs)
        for user,g,batch in records:
            job_number+=1;record=dict(job=job_number,user=user,model_version=seq,FL_round=20+fl,
                start_s=at,ready_s=at+float(times[user]),MACs=float(macs[user]),batch_size=len(batch),samples=batch)
            cached[user]=dict(gradient=g,**record);all_jobs.append(record)
            write(ad/'jobs'/f'{job_number:05d}.json',record)
        return compute
    def drain(at):
        return max([at,*[r['ready_s'] for r in cached.values()]])
    for _ in range(3000):
        if fl==180 and q.metrics(t)['procedure_completed']==45:break
        site=min(fl//36,4);prior_site=clock.sites[-1]['site'] if clock.sites else None
        if prior_site!=site:cached.clear()
        t=clock.enter(site,t,q,fl);q.sync(fl,t,occupied)
        arrived={rid for rid,r in q.items.items() if r['arrival_s'] is not None}
        if arrived!=seen:
            end=drain(t)
            if end>t:clock.wait(t,end);t=end
            cached.clear();seen=arrived
        kind,ids=q.decision(t,fl)
        if kind=='wait':clock.wait(t,ids);t=ids;continue
        assert kind in ['fl','fu']
        seq_start=seq;start=t;current_cost=np.zeros(10);local=[];selection=[]
        head=DEPLOYMENT['sites'][site]['head_zero_based'];config=plan['config']
        if kind=='fu':
            end=drain(t)
            if end>t:clock.wait(t,end);t=end;start=t
            cached.clear();engine.start_work()
            _,updates,local=fu_vectors(engine,theta,plan,q,t,fl,'fu',ids,None,{},False)
            cost=engine.finish_work();current_cost+=np.array(cost['device_macs'])
            _,ct=physical.compute_cost(current_cost);selection=list(map(int,np.flatnonzero(np.any(updates,axis=1))))
            wait=float(ct.max())
            if async_mode:vector,rm=radio.ota(updates,channels[site],config,head,500000+seq);compute_upload=wait+rm['uplink_s']
            else:vector,rm=radio.digital(updates,channels[site],config,head,ct);compute_upload=rm['compute_upload_finish_s']
        elif async_mode:
            if not cached:new_jobs(list(range(10)),t)
            ready=max(t,min(r['ready_s'] for r in cached.values()));wait=ready-t
            selection=[u for u,r in cached.items() if r['ready_s']<=ready+1e-12]
            updates=np.zeros((10,len(theta)-1));lr=base.core.lr_at(21+fl)
            versions={u:cached[u]['model_version'] for u in selection}
            for u in selection:updates[u]=-lr*cached[u]['gradient']/len(selection)
            vector,rm=radio.ota(updates,channels[site],config,head,500000+seq)
            rm.update(selected_model_versions=versions,staleness={u:seq-v for u,v in versions.items()},
                selected_job_ids=[cached[u]['job'] for u in selection],selection_rule='all currently ready; wait only if none ready')
            compute_upload=wait+rm['uplink_s']
            for u in selection:del cached[u]
        else:
            new_jobs(list(range(10)),t);_,ct=physical.compute_cost(current_cost)
            updates=np.stack([-base.core.lr_at(21+fl)*cached[u]['gradient']/10 for u in range(10)])
            selection=list(range(10));vector,rm=radio.digital(updates,channels[site],config,head,ct)
            compute_upload=rm['compute_upload_finish_s'];cached.clear()
        comm,down=communication(rm);end=t+compute_upload+down
        theta=update_theta(theta,vector)
        fl+=int(kind=='fl');q.last_uplink_s=rm['uplink_s'];q.commit(ids,t,end)
        t=end;seq+=1;occupied+=rm['uplink_s'];rf+=rm['energy_bound_j']
        # Start next local jobs immediately AFTER selected-device model broadcast.
        # No unused speculative job at region/mission boundary.
        if async_mode and kind=='fl' and fl<180 and fl%36!=0:new_jobs(selection,t)
        timing=dict(compute_s=compute_upload-rm['uplink_s'],uplink_s=rm['uplink_s'],downlink_s=down,
            duration_s=end-start,policy='Actual pipeline readiness then OTA' if async_mode else 'ISCC-P2 max_i(compute_i+upload_i), then broadcast')
        cost=dict(device_macs=current_cost.tolist(),scope='All jobs started in this transaction, including next async jobs; discarded jobs still charged.')
        audit=dict(attempt=seq,transaction=seq,status='transmitted',kind=kind,site=site,FU_request_ids=ids,
            FL_round=20+fl,compute=cost,radio=rm,timing=timing,local=local,selected_users=selection)
        write(ad/'work_attempts'/f'{seq:04d}.json',audit)
        due=(fl!=last_fl and fl%5==0) or any(q.items[i]['completed_s']==end for i in ids)
        ev=None
        if due:
            ev=base.clip.stream.evaluation(engine,theta,refprob,q,t);last_fl=fl
            if fl%5==0:
                ok=q.metrics(t)['procedure_completed']==45 and ev['retained']['mR']>=threshold and ev['forget']['KL_retrain_to_candidate']<=.01 and ev['worst_request_KL']<=.01
                if not hits or hits[-1]['fl_round']!=20+fl:hits=hits+[dict(fl_round=20+fl,time_s=t)] if ok else []
                elif not ok:hits=[]
                if len(hits)>=3 and target is None:target=dict(confirmation_FL_round=20+fl,mission_clock_s=t,first_pass=hits[0])
            write(ad/'evaluations'/f'{seq:04d}.json',point(t,ev))
        row=dict(**point(t,ev),kind=kind,FU_request_ids=ids,radio=rm,local=local,site=site,start_s=start,end_s=end,timing=timing)
        write(ad/'transactions'/f'{seq:04d}.json',row)
        if seq%20==0:state(stage='training',case=CURRENT_CASE['id'],dataset=engine.data.name if hasattr(engine.data,'name') else dest.name,method=name,FL_round=20+fl,transaction=seq)
    else:raise RuntimeError('workload did not finish')
    assert fl==180 and all(j['ready_s']<=t+1e-8 for j in all_jobs)
    final=base.clip.stream.evaluation(engine,theta,refprob,q,t)
    timeline=clock.finish(t,q,fl);write(ad/'mission_timeline.json',timeline)
    write(ad/'service_completion.json',point(t,final))
    mia,_=engine.mia_scores(theta);refmia=read(dest/'reference_mia.json');mia['auc_gap_vs_retrain']=abs(mia['loss_score_auc']-refmia['loss_score_auc'])
    write(ad/'mia.json',mia);base.core.save_theta(ad/'final.pt',theta,20+fl)
    write(ad/'request_records.json',list(q.items.values()))
    write(ad/'manifest.json',dict(status='completed_200FL_and_requests',**point(t,final)))


def validate(ad,requests,geo,config,cap):
    tx=[read(p) for p in sorted((ad/'transactions').glob('*.json'))];timeline=read(ad/'mission_timeline.json')
    req=read(ad/'request_records.json');work=[read(p) for p in sorted((ad/'work_attempts').glob('*.json'))]
    assert len(req)==45 and sum(t['kind']=='fl' for t in tx)==180
    assert sum(len(t['FU_request_ids']) for t in tx)==sum(r['required_steps'] for r in requests)
    for r in req:
        serving=[t for t in tx if r['id'] in t['FU_request_ids']]
        assert len(serving)==r['required_steps']==r['steps'];assert math.isclose(serving[-1]['end_s'],r['completed_s'])
    for t in tx:
        m=t['radio'];assert m['max_power_w']<=config['power_limit_w']*(1+1e-7)
        assert m['conditional_rmse']<=m['epsilon']*(1+1e-8)
        assert math.isclose(t['end_s']-t['start_s'],t['timing']['duration_s'],abs_tol=1e-9)
    for s in timeline['sites']:
        rows=[t for t in tx if t['site']==s['site']];wait=sum(w['duration_s'] for w in timeline['waits'] if w['site']==s['site'])
        assert len([t for t in rows if t['kind']=='fl'])==36
        assert math.isclose(sum(t['timing']['duration_s'] for t in rows)+wait,s['service_duration_s'],abs_tol=1e-8)
    for w in work:
        assert np.all(np.array(w['compute']['device_macs'])>=0)
        assert all(not r['clean_proxy_used'] and not r['request_teacher_used'] for r in w['local'])
    jobs=[read(p) for p in (ad/'jobs').glob('*.json')]
    selected=[j for t in tx for j in t['radio'].get('selected_job_ids',[])]
    assert len(selected)==len(set(selected))
    lookup={r['job']:r for r in jobs}
    # Every used async cache finishes before its transmission, and no cache
    # computed before an intervening FU/deletion release is ever applied.
    for t in tx:
        for j in t['radio'].get('selected_job_ids',[]):
            r=lookup[j];assert r['ready_s']<=t['start_s']+t['timing']['compute_s']+1e-8
            arrivals=[q['arrival_s'] for q in req if q['arrival_s'] is not None and r['start_s']<q['arrival_s']<=t['start_s']]
            assert not arrivals,(r,t['sequence'],arrivals)
    v=dict(passed=True,FL_rounds=200,FU_steps=sum(r['required_steps'] for r in req),FU_requests=45,transactions=len(tx),local_jobs=len(jobs),
        network_cap_overruns=sum(t['radio']['network_cap_overrun'] for t in tx),round_deadline_overruns=sum(t['radio']['round_deadline_overrun'] for t in tx),
        per_update_energy_overruns=sum(t['radio']['per_update_energy_overrun'] for t in tx),max_staleness=max([0,*[v for t in tx for v in t['radio'].get('staleness',{}).values()]]))
    return v,timeline,tx


def energy_ledger(dd,method,geo):
    import event_energy
    e=event_energy.energy;ad=dd/method;timeline=read(ad/'mission_timeline.json')
    work=[read(p) for p in sorted((ad/'work_attempts').glob('*.json'))]
    compute=sum(float(e.compute_cost(r['compute']['device_macs'])[0].sum()) for r in work)
    compute+=sum(e.PARAMETERS['communication']['head_processing_w']*r['radio']['uplink_s'] for r in work)
    comm=sum(communication(r['radio'])[0] for r in work)
    active=sum(s['service_duration_s'] for s in timeline['sites']);hover=10*float(e.propulsion_power(0))*active
    ev=read(ad/'manifest.json')['evaluation'];req=read(ad/'request_records.json');lat=[r['completed_s']-r['arrival_s'] for r in req]
    return dict(method=method,total_energy_j=compute+comm+hover,energy_computing_j=compute,energy_communication_j=comm,energy_hover_j=hover,
        delay_s=active,total_delay_s=active,mission_delay_s=timeline['mission_end_s'],mR=ev['retained']['mR'],loss=ev['retained']['loss'],
        forget_KL=ev['forget']['KL_retrain_to_candidate'],worst_request_KL=ev['worst_request_KL'],
        FU_mean_completion_latency_s=float(np.mean(lat)),FU_p95_completion_latency_s=float(np.percentile(lat,95)),
        energy_scope='Active FL/FU comm+compute+hover only; travel/collection/preflight excluded')


def run_case(cid,method):
    global BASE,BATCH,DEPLOYMENT,CURRENT_CASE
    register();case=next(c for c in cases() if c['id']==cid);CURRENT_CASE=case;BATCH=int(case['value']) if case['axis']=='batch' else 16
    lib.BATCH=BATCH;lib.ACTIVE='fedavg_clip_gaft';lib.infra.extra.Data=lib.infra.VideoData
    lib.adaptive.link(lib.adaptive.SOURCE/'prepared/airscape_seed0',OUT/'prepared/airscape_seed0')
    _,base=lib.infra.prior.load_base()
    import worker,accounting,event_energy
    lib.infra.ORIGINAL_INSTRUMENTED=accounting.instrumented_engine;accounting.instrumented_engine=lib.infra.controlled_engine
    event_energy.method_energy=energy_ledger
    def maker(b,deployment,radios,cap):
        global BASE,DEPLOYMENT
        BASE=b;DEPLOYMENT=deployment;return train_arm
    run=base.repeat.clone_function(worker.run,[
        ("opt=base.base.ARMS['proposed_fu_guard_cliperase' if method=='heuristic_ova' else method]",'opt=dict(method=method,scope="paper-core transfer; see protocol.json")',1)],
        dict(OUT=OUT,METHODS=list(METHODS),EXTRA_DATASETS=['AirScape'],make_arm=maker,validate=validate,state=state))
    run(case,DATASETS,[method]);outcomes={ds:read(OUT/'cases'/cid/(ds.lower()+'_seed0')/method/'outcome.json')['status'] for ds in DATASETS}
    assert 'failed' not in outcomes.values(),outcomes
    print(dict(case=cid,method=method,statuses=outcomes),flush=True)


def verify():
    register();rows=[]
    for case in cases():
        for ds in DATASETS:
            for method in METHODS:
                ad=OUT/'cases'/case['id']/(ds.lower()+'_seed0')/method;o=read(ad/'outcome.json')
                assert o['status']=='completed',o
                v=read(ad/'verification.json');assert v['passed'];rows.append(dict(case=case['id'],dataset=ds,method=method,**v))
    result=dict(passed=True,completed_arms=len(rows),rows=rows);write(OUT/'verification.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['unit','case','run','verify']);p.add_argument('--case',default='baseline');p.add_argument('--method',choices=list(METHODS),default=next(iter(METHODS)));a=p.parse_args()
    if a.action=='unit':print(radio.unit());return
    if a.action=='case':run_case(a.case,a.method);return
    if a.action=='verify':print(verify());return
    register()
    for c in cases():
        for method in METHODS:subprocess.run([sys.executable,str(Path(__file__)),'case','--case',c['id'],'--method',method],cwd=ROOT,check=True)
    result=verify();write(OUT/'finished.json',dict(unix=time.time(),**result));state(stage='completed')


if __name__=='__main__':main()
