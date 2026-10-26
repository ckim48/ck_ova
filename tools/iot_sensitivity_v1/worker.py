"""One isolated scenario; same checkpoints and paired seeds for all methods."""
from common import *
import argparse, concurrent.futures, fcntl, math, multiprocessing, traceback
from types import SimpleNamespace
import numpy as np

def queues(base,bandwidth):
    class Joint(base.RoundQueue):
        def metrics(self,t):
            r=super().metrics(t);r['meaning']='All prescribed per-request FU steps and final broadcast completed; no deletion certificate.';return r
    class Serial(base.SerialRoundQueue):
        def __init__(self,requests,parallel=False,period_frames=5):
            super().__init__(requests,parallel,period_frames);self.period=period_frames*20480/bandwidth
        def metrics(self,t):return Joint.metrics(self,t)
    # Joint.metrics uses zero-argument super; do not call it on unrelated class.
    def serial_metrics(self,t):
        r=base.SerialRoundQueue.metrics(self,t);r['meaning']='All prescribed FU steps and final broadcast completed; no deletion certificate.';return r
    Serial.metrics=serial_metrics
    return SimpleNamespace(**dict(base.__dict__,RoundQueue=Joint,SerialRoundQueue=Serial))

def replay(base,requests,geometry,deployment,config):
    from mission_clock import MissionClock,queue_classes
    joint,serial=queue_classes(base);results=[]
    for parallel in [True,False]:
        for duration in [20480/config['symbol_rate_hz']+.163968, 3.]:
            q=joint(requests) if parallel else serial(requests,False,5)
            clock=MissionClock(geometry,deployment);t=fl=occupied=0
            for _ in range(2000):
                if fl==180 and q.metrics(t)['procedure_completed']==45:break
                site=min(fl//36,4);t=clock.enter(site,t,q,fl);q.sync(fl,t,occupied);kind,ids=q.decision(t,fl)
                if kind=='wait':clock.wait(t,ids);t=ids;continue
                assert all(q.items[r]['mission_site']==site for r in ids)
                q.last_uplink_s=.001;q.commit(ids,t,t+duration)
                fl+=int(kind in ['fl','joint']);t+=duration;occupied+=.001
            else:raise AssertionError('Schedule did not terminate')
            timeline=clock.finish(t,q,fl)
            assert all(r['steps']==r['required_steps'] for r in q.items.values())
            results.append(dict(parallel=parallel,duration=duration,mission_end_s=timeline['mission_end_s']))
    return results

def validate(ad,requests,geo,config,cap):
    from mission_clock import MissionClock
    manifest=read(ad/'manifest.json');assert manifest['completed_FL_rounds']==200
    tx=[read(p) for p in sorted((ad/'transactions').glob('*.json'))]
    work={r['transaction']:r for r in (read(p) for p in (ad/'work_attempts').glob('*.json'))}
    req=read(ad/'request_records.json');timeline=read(ad/'mission_timeline.json')
    assert len(req)==45 and sum(len(r['FU_request_ids']) for r in tx)==sum(r['required_steps'] for r in requests)
    for r in req:
        serving=[t for t in tx if r['id'] in t['FU_request_ids']]
        assert len(serving)==r['steps']==r['required_steps']
        assert math.isclose(serving[-1]['end_s'],r['completed_s'])
    for s in timeline['sites']:
        rows=[t for t in tx if t['site']==s['site']]
        assert sum(t['kind'] in ['fl','joint'] for t in rows)==36
        assert math.isclose(rows[-1]['end_s'],s['departure_s'])
        wait=sum(w['duration_s'] for w in timeline['waits'] if w['site']==s['site'])
        assert math.isclose(sum(t['timing']['duration_s'] for t in rows)+wait,s['service_duration_s'])
    for row in tx:
        m=row['radio'];a=work[row['sequence']]
        timing=MissionClock.transaction_timing(a['compute'],m)
        assert math.isclose(row['end_s']-row['start_s'],timing['duration_s'],abs_tol=1e-9)
        assert m['uplink_s']<=config['round_uplink_deadline_s']+1e-10
        assert m['max_user_energy_j']<=config['available_energy_j']*(1+2e-6)
        assert m['max_power_w']<=config['power_limit_w']*(1+2e-6)
        assert m['conditional_rmse']<=m['epsilon']*(1+3e-5)+1e-9
        if ad.name!='sequential_digital_periodic':assert m['energy_bound_j']<=cap*(1+2e-6)
        if row['kind']=='joint':assert a['FL_clients_apply_s']==a['FU_clients_apply_s']
    assert math.isclose(timeline['mission_end_s'],sum(s['hover_duration_s'] for s in timeline['sites'])+sum(l['duration_s'] for l in timeline['legs']))
    assert math.isclose(timeline['mission_end_s']-timeline['service_end_s'],timeline['legs'][-1]['duration_s'])
    return dict(passed=True,FL_rounds=200,FU_requests=45,FU_steps=sum(r['required_steps'] for r in req),transactions=len(tx)),timeline,tx

def make_arm(base,deployment,radios,cap):
    import event_arm
    extra=[("radio=base.parent.guard if name=='proposed_fu_guard_cliperase' else base.SequentialRadio if name=='sequential_digital_periodic' else base.parent.common","radio=Radios[name]",1),
           ("if name!='sequential_digital_periodic':assert rm['energy_bound_j']<=.1*(1+2e-6)","if name!='sequential_digital_periodic':assert rm['energy_bound_j']<=NETWORK_CAP*(1+2e-6)",1)]
    maker=base.repeat.clone_function(event_arm.make_arm,[
        ('return base.repeat.clone_function','edits.extend(EXTRA_EDITS)\n    return base.repeat.clone_function',1),
        ('deployment=deployment,state=state))','deployment=deployment,state=state,Radios=RADIOS,NETWORK_CAP=CAP))',1)],
        dict(EXTRA_EDITS=extra,RADIOS=radios,CAP=cap))
    return maker(base,deployment,state)

def run(case,datasets,only=None,preflight=False):
    event,base=load_base()
    import event_energy,overrun_radio
    from heuristic import build,pool_init
    from accounting import instrumented_engine
    dest=OUT/'cases'/case['id'];dest.mkdir(parents=True,exist_ok=True)
    geometry=read(GEOMETRY/'mission_spec.json');deployment=read(GEOMETRY/'deployment.json')
    template=read(SOURCE/'capera_seed0/plan.json');req=read(SOURCE/'capera_seed0/requests.json')
    params=copy.deepcopy(event_energy.PARAMETERS)
    cap,channels,collection_ok=configure(case,template,req,geometry,deployment,params,np.load(GEOMETRY/'channels_by_site.npy'))
    base=queues(base,template['config']['symbol_rate_hz'])
    event_energy.energy.PARAMETERS=params;event_energy.PARAMETERS=params;event_energy.DEPLOYMENT=dest/'deployment.json'
    write(dest/'deployment.json',deployment);write(dest/'mission_spec.json',geometry);write(dest/'settings.json',dict(case=case,config=template['config'],energy_model=params,network_cap_j=cap,collection_feasible=collection_ok))
    checks=replay(base,req,geometry,deployment,template['config']);write(dest/'preflight.json',dict(passed=True,schedules=checks))
    if preflight:return
    import torch
    assert torch.cuda.is_available();torch.set_num_threads(1)
    gpu=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX)
    p=base.base.parent.common.base;p.CAP=cap
    c=template['config'];p.COUNTS=tuple(n for n in p.COUNTS if n*20480/c['symbol_rate_hz']<=c['round_uplink_deadline_s']+1e-10 and n*128/c['symbol_rate_hz']<=c['coherence_time_s']+1e-10)
    assert p.COUNTS
    p.POOL=concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),initializer=pool_init,initargs=(cap,))
    overrun_radio.ORIGINAL_CAP_J=cap
    radios={m:base.base.parent.common for m in METHODS}
    radios.update(proposed_fu_guard_cliperase=base.base.parent.guard,heuristic_ova=build(base,geometry,channels,cap),sequential_digital_periodic=overrun_radio.build(base))
    arm=make_arm(base,deployment,radios,cap);results=[]
    original_prefix_cost=event_energy.prefix_cost
    try:
        for dataset in datasets:
            src=SOURCE/(dataset.lower()+'_seed0')
            if not src.exists():src=OUT/'prepared'/(dataset.lower()+'_seed0')
            dd=dest/(dataset.lower()+'_seed0');dd.mkdir(parents=True,exist_ok=True)
            plan=read(src/'plan.json');requests=read(src/'requests.json')
            configure(case,plan,requests,copy.deepcopy(read(GEOMETRY/'mission_spec.json')),copy.deepcopy(read(GEOMETRY/'deployment.json')),copy.deepcopy(event_energy.PARAMETERS),np.load(GEOMETRY/'channels_by_site.npy'))
            write(dd/'plan.json',plan);write(dd/'requests.json',requests)
            for name in ['feature_cache','initial20','proxy20','reference200','common_prefix_cost.json','reference_metrics.json','reference_mia.json']:
                target=dd/name
                if not target.exists():target.symlink_to(os.path.relpath(src/name,dd),target_is_directory=(src/name).is_dir())
            if not collection_ok:
                for method in METHODS:
                    write(dd/method/'outcome.json',dict(status='infeasible_collection_power',case=case['id'],dataset=dataset,method=method))
                continue
            if dataset in EXTRA_DATASETS:
                from prepare_extra import Data,native_engine,native_base
                data=Data(dataset);Engine=native_engine(base,data.caption_count)
                arm=make_arm(native_base(base,data.caption_count),deployment,radios,cap)
                event_energy.prefix_cost=base.repeat.clone_function(original_prefix_cost,[('+5*text','+NCAP*text',1)],{'NCAP':data.caption_count})
            else:
                data=event.recover.Data(dataset);Engine=base.Engine
                event_energy.prefix_cost=original_prefix_cost
            payload=torch.load(dd/'feature_cache/features.pt',map_location='cpu',weights_only=False)
            engine=instrumented_engine(Engine)(data,plan,payload)
            warm=base.core.load_theta(dd/'initial20/final.pt',engine.device);proxy=base.core.load_theta(dd/'proxy20/final.pt',engine.device)
            ref=base.core.load_theta(dd/'reference200/final.pt',engine.device);engine.set_teacher(warm);_,refprob=engine.evaluate(ref,None)
            for method in (only or METHODS):
                ad=dd/method
                if (ad/'outcome.json').exists():continue
                if (ad/'started.json').exists():raise RuntimeError('Incomplete prior attempt: preserve and review '+str(ad))
                start=time.time();state(stage='running',case=case['id'],dataset=dataset,method=method)
                outcome=dict(case=case['id'],dataset=dataset,method=method)
                try:
                    opt=base.base.ARMS['proposed_fu_guard_cliperase' if method=='heuristic_ova' else method]
                    arm(engine,plan,requests,geometry,channels,method,opt,read(dd/'common_prefix_cost.json'),warm,proxy,refprob,read(dd/'reference_metrics.json')['retained_mR_threshold'],dd,0)
                    manifest=read(ad/'manifest.json')
                    if manifest['status']!='completed_200FL_and_requests':outcome.update(status=manifest['status'])
                    else:
                        checked,timeline,tx=validate(ad,requests,geometry,plan['config'],cap)
                        write(ad/'verification.json',checked);m=event_energy.method_energy(dd,method,geometry)
                        m.update(FL_FU_completion_delay_s=timeline['service_end_s'],FL_FU_active_service_s=sum(s['service_duration_s'] for s in timeline['sites']),
                            hover_time_s=sum(s['hover_duration_s'] for s in timeline['sites']),return_flight_s=timeline['legs'][-1]['duration_s'],
                            uplink_s=sum(t['radio']['uplink_s'] for t in tx),adaptation_RF_energy_j=sum(t['radio']['energy_bound_j'] for t in tx),
                            original_network_tx_cap_j=cap,network_cap_overrun_transactions=sum(t['radio']['energy_bound_j']>cap*(1+2e-6) for t in tx),
                            network_cap_total_excess_j=sum(max(0.,t['radio']['energy_bound_j']-cap) for t in tx),
                            delay_scope='FL_FU completion: takeoff to all required work and broadcasts complete, includes preceding travel and collection; mission: also return; total: also preflight prefix; active service excludes travel and collection.')
                        write(ad/'energy_and_delay.json',m);outcome.update(status='completed',metrics=m)
                except Exception as e:
                    outcome.update(status='failed',error=repr(e));write(ad/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
                outcome['wall_seconds']=time.time()-start;write(ad/'outcome.json',outcome);results.append(outcome)
            del engine,data,payload,refprob
            import gc;gc.collect();torch.cuda.empty_cache()
        write(dest/'finished.json',dict(unix=time.time(),outcomes=[{k:v for k,v in r.items() if k!='metrics'} for r in results]))
    finally:p.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',default='baseline');parser.add_argument('--datasets',nargs='+',default=['RSITMD','CapERA']);parser.add_argument('--methods',nargs='+');parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    run(next(c for c in cases() if c['id']==args.case),args.datasets,args.methods,args.preflight)
