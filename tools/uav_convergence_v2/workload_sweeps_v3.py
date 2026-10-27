"""New workload/ARQ conditions with source-frozen methods and actual clocks.

Same methods, data and VLM objectives. FL budgets stop the existing cosine
schedule earlier; every condition still serves the identical45 deletion requests.
"""
import argparse
import copy
import hashlib
import inspect
import linecache
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import paper_uav_benchmarks as paper
import literature_baselines as lib
from adaptive_fu import ROOT,read,write,sha

OUT=ROOT/'results/workload_sweeps_v3_20261028'
METHODS={lib.PROPOSED:'Proposed',**paper.METHODS}
DATASETS=paper.DATASETS
AXES={
 'request_bursts':dict(label='FU requests per release',values=[45,15,9,5,3],display_values=[1,3,5,9,15],base=9),
 'fl_rounds':dict(label='FL-round budget',values=[100,125,150,175,200],base=200),
 'packet_loss':dict(label='Downlink packet loss (%)',values=[0,.1,.2,.3,.4],display_values=[0,10,20,30,40],base=0)}
CASE=None;ADAPT=180;SITE_FL=36;TOTAL=200;LOSS=0.;PHYSICAL=None


def case_id(axis,v):return 'baseline' if v==AXES[axis]['base'] else axis+'_'+str(v).replace('.','p')

def cases():return [dict(id='control',axis=None,value=None)]+[dict(id=case_id(a,v),axis=a,value=v) for a,s in AXES.items() for v in s['values'] if v!=s['base']]

def state(dest=None,**kw):write((Path(dest) if dest else OUT)/'status.json',dict(unix=time.time(),pid=os.getpid(),**kw))


def register():
    path=OUT/'protocol.json'
    if path.exists():
        p=read(path)
        for k,v in p['source_sha256'].items():assert sha(ROOT/k)==v,k
        return p
    old=paper.register();sources=dict(old['source_sha256'])
    for name in ['paper_uav_drain_fix.py','paper_uav_charts.py',Path(__file__).name]:
        f=Path(__file__).with_name(name);sources[str(f.relative_to(ROOT))]=sha(f)
    p=dict(registered_unix=time.time(),methods=METHODS,datasets=DATASETS,axes=AXES,cases=cases(),new_training_arms=70,
      scope='6 zero-change control arms +64 new arms.8 identical prior Proposed request-grouping arms reused, never rescaled. Main plots combine existing uplink/downlink with new request grouping,FL budget,packet erasure. All previous flat/losing outcomes retained.',
      FL_budget='20 common warmup rounds +80/105/130/155/180 mission FL updates; same original cosine LR indexed by absolute round, truncated at specified budget. Five sites get equal mission FL counts. Keep45 requests and their forget cohort; scale each original within-site release ordinal by floor(offset*new_site_rounds/36). All deletion work completes before departure. Reference200 remains reporting-only.',
      grouping='Existing shared45-request release-group intervention; group counts45/15/9/5/3 give1/3/5/9/15 requests per release. Same individual sample cohorts and5FU cycles. No quality-based scenario selection.',
      ARQ='Exogenous aggregate multicast packet erasure probability0/0.1/0.2/0.3/0.4. Every full model+control broadcast retries until success; exact retry draws/counts recorded. One Bernoulli outcome represents successful delivery to all intended receivers. Shared deterministic uniforms per transaction and retry across probability settings and methods; no extra delay scaling or outcome rejection. All failed attempts consume broadcast time,PA/circuit energy and hover. No extra model update for retries. Uplink CSI/error model unchanged. Simplified independent packet erasures, not calibrated channel measurements.',
      fixed='Same10UAV, datasets,cachedfeatures,warm20,proxy/reference,45requests,radio limits and energy coefficients, VLM objectives and method mechanics. BATCH16. Seed0 development study; no confidence or official benchmark claims.',
      integrity='Freeze this source before training. Preserve failed outcomes. All raw traces/checkpoints local; publish code,charts,plot values and aggregate checks only.',source_sha256=sources)
    write(path,p);return p


def configure(case,plan,requests,geometry,deployment,params,channels):
    import common
    result=common.configure(case,plan,requests,geometry,deployment,params,channels)
    if case['axis']=='fl_rounds':
        n=(int(case['value'])-20)//5;plan['rounds']=int(case['value'])
        plan['workload_budget_note']='Same cosine schedule truncated at declared totalFL; same deletion cohort; within-site release ordinals proportionally compressed.'
        for r in requests:
            offset=r['release_after_adaptation_FL']-r['mission_site']*36
            release=r['mission_site']*n+int(math.floor(offset*n/36))
            r.update(release_after_adaptation_FL=release,release_before_total_FL_round=21+release)
    if case['axis']=='packet_loss':params['communication']['packet_erasure_probability']=case['value']
    return result


def attach_arq(meta,sequence):
    active=meta['uplink_s']>0 or meta.get('broadcast_required',False)
    draws=[]
    if active:
        rng=np.random.default_rng(8200000+sequence)
        for _ in range(1000):
            u=float(rng.random());draws.append(u)
            if u>=LOSS:break
        else:raise RuntimeError('ARQ safety limit; preserve failed workload')
    meta.update(broadcast_attempts=len(draws),broadcast_erasure_probability=LOSS,broadcast_uniforms=draws,
        broadcast_trial_seed=8200000+sequence,broadcast_scope='Whole model plus control; one success delivers to all intended peers')


def communication(meta):
    p=PHYSICAL.PARAMETERS['communication'];up=meta['uplink_s']
    energy=meta['energy_bound_j']/p['pa_efficiency']+up*(9*p['tx_circuit_w']+p['rx_circuit_w'])
    active=up>0 or meta.get('broadcast_required',False)
    count=meta.get('broadcast_attempts',int(active))
    down=count*(p['update_coordinates']*p['broadcast_bits_per_coordinate']+p['control_bits_per_update'])/p['downlink_rate_bps']
    energy+=down*(p['downlink_power_w']/p['pa_efficiency']+p['tx_circuit_w']+9*p['rx_circuit_w'])
    return energy,down


def make_clock():
    from mission_clock import MissionClock
    class Clock(MissionClock):
        def close_site(self,t,q,fl):
            s=self.sites[-1];site=s['site'];assert fl==(site+1)*SITE_FL
            region=[r for r in q.items.values() if r['mission_site']==site]
            assert all(r['arrival_s'] is not None and r['steps']==r['required_steps'] for r in region)
            s.update(service_end_s=t,departure_s=t,service_duration_s=t-s['service_start_s'],hover_duration_s=t-s['arrival_s'],completed_requests=len(region),FL_updates=SITE_FL)
        def finish(self,t,q,fl):
            assert len(self.sites)==5 and fl==ADAPT;self.close_site(t,q,fl);end=t
            self.legs.append(dict(from_site=4,to_site=0,start_s=t,end_s=t+self.durations[4],duration_s=self.durations[4]));t+=self.durations[4]
            assert np.isclose(t,sum(s['hover_duration_s'] for s in self.sites)+sum(l['duration_s'] for l in self.legs))
            return dict(policy='Depart on completion of declared FL budget and all regional FU work.',sites=self.sites,legs=self.legs,waits=self.waits,service_end_s=end,mission_end_s=t)
    return Clock


def queues(base,bandwidth):
    import worker
    base.repeat.clone_function=cached_clone
    b=worker.queues(base,bandwidth)
    for cls in [b.RoundQueue,b.SerialRoundQueue]:
        cls.decision=base.repeat.clone_function(cls.decision,[('fl < 180','fl < ADAPT',1)],dict(ADAPT=ADAPT))
    return b


def replay(base,requests,geometry,deployment,config):
    from mission_clock import queue_classes
    joint,serial=queue_classes(base);Clock=make_clock();results=[]
    for cls in [joint,serial]:
        q=cls(requests);clock=Clock(geometry,deployment);fl=0;t=occupied=0.
        for _ in range(2500):
            if fl==ADAPT and q.metrics(t)['procedure_completed']==45:break
            t=clock.enter(min(fl//SITE_FL,4),t,q,fl);q.sync(fl,t,occupied);kind,ids=q.decision(t,fl)
            if kind=='wait':clock.wait(t,ids);t=ids;continue
            assert all(q.items[r]['mission_site']==min(fl//SITE_FL,4) for r in ids)
            q.last_uplink_s=.01;q.commit(ids,t,t+.75);fl+=int(kind in ['fl','joint']);t+=.75;occupied+=.01
        else:raise AssertionError('preflight schedule')
        clock.finish(t,q,fl);results.append(dict(FL_rounds=20+fl,FU_cycles=sum(r['steps'] for r in q.items.values()),parallel=cls is joint))
    return results


def cached_clone(fn,edits,extra=None,live_globals=None):
    import textwrap
    source=textwrap.dedent(inspect.getsource(fn))
    for old,new,count in edits:
        assert source.count(old)==count,(fn.__name__,old,source.count(old),count)
        source=source.replace(old,new)
    namespace=live_globals if live_globals is not None else dict(fn.__globals__)
    if extra:namespace.update(extra)
    filename='/tmp/workload_clone_'+hashlib.sha256(source.encode()).hexdigest()+'.py'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    exec(compile(source,filename,'exec'),namespace)
    return namespace[fn.__name__]


def proposed_arm(base,deployment,radios,cap):
    fn=lib.make_arm(base,deployment,radios,cap)
    source=inspect.getsource(fn)
    edits=[('fl==180','fl==ADAPT',source.count('fl==180')),('fl//36','fl//SITE_FL',1),
        ("rm['broadcast_required']=bool(np.any(vector != 0))", "rm['broadcast_required']=bool(np.any(vector != 0))\n  attach_arq(rm,seq)",1),
        ("'completed_200FL_and_requests'","'completed_FL_budget_and_requests'",1)]
    return base.repeat.clone_function(fn,edits,dict(ADAPT=ADAPT,SITE_FL=SITE_FL,MissionClock=make_clock(),attach_arq=attach_arq,state=state))


def paper_arm():
    source=inspect.getsource(paper.train_arm)
    source=source.replace('from mission_clock import MissionClock,queue_classes','from mission_clock import queue_classes')
    source=source.replace('180','ADAPT_ROUNDS').replace('fl//36','fl//SITE_ROUNDS').replace('fl%36','fl%SITE_ROUNDS')
    old="    assert fl==ADAPT_ROUNDS and all(j['ready_s']<=t+1e-8 for j in all_jobs)"
    source=source.replace(old,"    terminal=max([t,*[j['ready_s'] for j in all_jobs]])\n    if terminal>t:clock.wait(t,terminal);t=terminal\n"+old)
    source=source.replace('        comm,down=communication(rm);','        attach_arq(rm,seq)\n        comm,down=communication(rm);')
    source=source.replace("'completed_200FL_and_requests'","'completed_FL_budget_and_requests'")
    source=source.replace('def train_arm(', 'def workload_arm(',1)
    filename='/tmp/workload_paper_'+hashlib.sha256(source.encode()).hexdigest()+'.py'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    paper.MissionClock=make_clock()
    paper.__dict__.update(ADAPT_ROUNDS=ADAPT,SITE_ROUNDS=SITE_FL,attach_arq=attach_arq,communication=communication,state=state)
    exec(compile(source,filename,'exec'),paper.__dict__)
    return paper.workload_arm


def validate(ad,requests,geo,config,cap):
    tx=[read(p) for p in sorted((ad/'transactions').glob('*.json'))];tl=read(ad/'mission_timeline.json');req=read(ad/'request_records.json')
    works={w['transaction']:w for w in [read(p) for p in (ad/'work_attempts').glob('*.json')]}
    assert read(ad/'manifest.json')['completed_FL_rounds']==TOTAL
    assert len(req)==45 and sum(t['kind'] in ['fl','joint'] for t in tx)==ADAPT
    assert sum(len(t['FU_request_ids']) for t in tx)==225
    for r in req:
        done=[t for t in tx if r['id'] in t['FU_request_ids']]
        assert len(done)==r['steps']==r['required_steps']==5 and math.isclose(done[-1]['end_s'],r['completed_s'])
    for s in tl['sites']:
        ts=[t for t in tx if t['site']==s['site']]
        assert sum(t['kind'] in ['fl','joint'] for t in ts)==SITE_FL
        wait=sum(w['duration_s'] for w in tl['waits'] if w['site']==s['site'])
        assert math.isclose(sum(t['timing']['duration_s'] for t in ts)+wait,s['service_duration_s'],abs_tol=1e-8)
    for t in tx:
        m=t['radio'];assert m['max_power_w']<=config['power_limit_w']*(1+2e-6)
        assert m['conditional_rmse']<=m['epsilon']*(1+3e-5)+1e-9
        assert math.isclose(communication(m)[1],t['timing']['downlink_s'],abs_tol=1e-10)
        draws=m['broadcast_uniforms'];assert len(draws)==m['broadcast_attempts']
        if draws:assert all(x<LOSS for x in draws[:-1]) and draws[-1]>=LOSS
        assert math.isclose(t['timing']['duration_s'],t['end_s']-t['start_s'],abs_tol=1e-8)
        if ad.name==lib.PROPOSED:
            assert m['energy_bound_j']<=cap*(1+2e-6) and m['max_user_energy_j']<=config['available_energy_j']*(1+2e-6)
            assert m['uplink_s']<=config['round_uplink_deadline_s']+1e-10
            if t['kind']=='joint':assert works[t['sequence']]['FL_clients_apply_s']==works[t['sequence']]['FU_clients_apply_s']
    jobs=[read(p) for p in (ad/'jobs').glob('*.json')]
    assert all(j['ready_s']<=tl['service_end_s']+1e-8 for j in jobs)
    v=dict(passed=True,FL_rounds=TOTAL,FU_requests=45,FU_steps=225,transactions=len(tx),
        broadcast_attempts=sum(t['radio']['broadcast_attempts'] for t in tx),broadcast_retries=sum(max(0,t['radio']['broadcast_attempts']-1) for t in tx),
        max_staleness=max([0,*[v for t in tx for v in t['radio'].get('staleness',{}).values()]]))
    return v,tl,tx


def reused(case,method):return case.startswith('request_bursts_') and method==lib.PROPOSED


def run_case(cid,group='all'):
    global CASE,ADAPT,SITE_FL,TOTAL,LOSS,PHYSICAL
    register();CASE=next(c for c in cases() if c['id']==cid);TOTAL=CASE['value'] if CASE['axis']=='fl_rounds' else 200
    ADAPT=TOTAL-20;SITE_FL=ADAPT//5;LOSS=CASE['value'] if CASE['axis']=='packet_loss' else 0.
    lib.OUT=OUT;lib.BATCH=16;paper.BATCH=16;paper.CURRENT_CASE=CASE;paper.OUT=OUT
    lib.infra.extra.Data=lib.infra.VideoData;lib.infra.OUT=OUT;lib.adaptive.TASKROOT=OUT
    lib.adaptive.link(lib.adaptive.SOURCE/'prepared/airscape_seed0',OUT/'prepared/airscape_seed0')
    _,base=lib.infra.prior.load_base();base.repeat.clone_function=cached_clone
    import worker,accounting,event_energy,heuristic
    PHYSICAL=event_energy.energy;PHYSICAL.communication=communication;paper.communication=communication
    event_energy.method_energy=paper.energy_ledger
    lib.infra.ORIGINAL_INSTRUMENTED=accounting.instrumented_engine;accounting.instrumented_engine=lib.infra.controlled_engine;heuristic.pool_init=lib.infra.pool_init
    def maker(b,deployment,radios,cap):
        paper.BASE=b;paper.DEPLOYMENT=deployment
        radios[lib.PROPOSED]=lib.adaptive.radio(b)
        # Proposed builds its sampler at arm creation; keep ACTIVE pinned here.
        lib.ACTIVE=lib.PROPOSED;lib.adaptive.ACTIVE_METHOD=lib.PROPOSED
        prop=proposed_arm(b,deployment,radios,cap);other=paper_arm()
        def route(*args):
            method=args[5]
            lib.ACTIVE=lib.PROPOSED if method==lib.PROPOSED else 'fedavg_clip_gaft'
            return (prop if method==lib.PROPOSED else other)(*args)
        return route
    def options(b,m):return lib.options(b,m) if m==lib.PROPOSED else dict(method=m,scope='Original paper-core transfer; shared workload/ARQ intervention')
    run=base.repeat.clone_function(worker.run,[
        ("opt=base.base.ARMS['proposed_fu_guard_cliperase' if method=='heuristic_ova' else method]",'opt=options(base,method)',1),
        ("'completed_200FL_and_requests'","'completed_FL_budget_and_requests'",1)],
        dict(OUT=OUT,METHODS=list(METHODS),EXTRA_DATASETS=['AirScape'],make_arm=maker,validate=validate,state=state,options=options,queues=queues,replay=replay,configure=configure))
    chosen=list(METHODS) if group=='all' else [lib.PROPOSED] if group=='proposed' else list(paper.METHODS)
    new=[m for m in chosen if not reused(cid,m)]
    run(CASE,DATASETS,new,preflight=not new)
    if lib.PROPOSED in chosen and reused(cid,lib.PROPOSED):
        for ds in DATASETS:
            source=lib.adaptive.OUT/'cases'/cid/(ds.lower()+'_seed0');dest=OUT/'cases'/cid/source.name
            for name in ['plan.json','requests.json']:
                if (dest/name).exists():assert read(dest/name)==read(source/name)
                else:lib.adaptive.link(source/name,dest/name)
            lib.adaptive.link(source/lib.PROPOSED,dest/lib.PROPOSED)
    statuses={ds+':'+m:read(OUT/'cases'/cid/(ds.lower()+'_seed0')/m/'outcome.json')['status'] for ds in DATASETS for m in chosen}
    assert 'failed' not in statuses.values(),statuses
    print(dict(case=cid,statuses=statuses),flush=True)


def unit():
    global CASE,ADAPT,SITE_FL,TOTAL,LOSS,PHYSICAL
    _,base=lib.infra.prior.load_base();base.repeat.clone_function=cached_clone
    import worker,common,event_energy
    PHYSICAL=event_energy.energy
    original=read(common.SOURCE/'capera_seed0/requests.json');plan=read(common.SOURCE/'capera_seed0/plan.json')
    geometry=read(common.GEOMETRY/'mission_spec.json');deployment=read(common.GEOMETRY/'deployment.json')
    for c in cases():
        TOTAL=c['value'] if c['axis']=='fl_rounds' else 200;ADAPT=TOTAL-20;SITE_FL=ADAPT//5
        req=copy.deepcopy(original);params=copy.deepcopy(event_energy.PARAMETERS)
        configure(c,copy.deepcopy(plan),req,copy.deepcopy(geometry),copy.deepcopy(deployment),params,np.ones((5,10,10)))
        replay(queues(base,1e6),req,geometry,deployment,plan['config'])
    for p in [0.,.1,.2,.3,.4]:
        LOSS=p;counts=[]
        for k in range(3000):
            m=dict(uplink_s=.01,energy_bound_j=.001,broadcast_required=True);attach_arq(m,k);counts.append(m['broadcast_attempts'])
        assert abs(np.mean(counts)-1/(1-p))<.06
    LOSS=0.;m=dict(uplink_s=.01,energy_bound_j=.001);attach_arq(m,0);assert m['broadcast_attempts']==1
    lib.ACTIVE=lib.PROPOSED;lib.adaptive.ACTIVE_METHOD=lib.PROPOSED
    proposed_arm(base,deployment,{lib.PROPOSED:lib.adaptive.radio(base)},.1)
    paper_arm()
    result=dict(passed=True,dynamic_arm_source_available=True,all_release_schedules_complete_same45_requests=True,ARQ_sample_mean_matches_geometric_expectation=True,zero_loss_single_broadcast=True)
    write(OUT/'unit_checks.json',result);print(result)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['unit','case']);p.add_argument('--case',default='control');p.add_argument('--group',choices=['all','proposed','papers'],default='all');a=p.parse_args()
    unit() if a.action=='unit' else run_case(a.case,a.group)


if __name__=='__main__':main()
