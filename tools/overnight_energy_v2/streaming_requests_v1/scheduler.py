"""Causal scheduling on an uplink-service clock (local computation excluded)."""
import math,copy
import numpy as np
STEPS=5
FL_STEPS=180  # plus the common twenty-round initial model
FRAME_S=20480/1e6
HORIZON=FL_STEPS*FRAME_S

def make_requests(cohort,interval_frames=20):
    rng=np.random.default_rng(231023)
    chunks={u:np.array_split(rng.permutation(ids),9) for u,ids in enumerate(cohort) if ids}
    jitter=np.random.default_rng(9123).integers(-3,4,9);jitter[0]=0
    result=[]
    for b in range(9):
        at=max(0,b*interval_frames+int(jitter[b]))*FRAME_S
        for u,parts in chunks.items():
            ids=sorted(map(int,parts[b]));assert len(ids)>=2
            result.append(dict(id=f'b{b:02d}_u{u}',burst=b,user=u,images=ids,
                arrival_s=at,deadline_s=at+20*FRAME_S,required_steps=STEPS))
    assert len(result)==45
    assert len({i for r in result for i in r['images']})==sum(map(len,cohort))
    return result

class Queue:
    def __init__(self,requests,parallel,period_frames=5):
        self.items={r['id']:dict(copy.deepcopy(r),steps=0,first_service_s=None,completed_s=None) for r in requests}
        self.parallel=parallel;self.period=period_frames*FRAME_S;self.batch=[]
    def arrived(self,t):return [r for r in self.items.values() if r['arrival_s']<=t+1e-10]
    def pending(self,t):return [r for r in self.arrived(t) if r['steps']<STEPS]
    def decision(self,t,fl):
        pending=self.pending(t)
        if self.parallel:
            if fl<FL_STEPS or pending:return ('joint' if pending else 'fl'),[r['id'] for r in pending]
        else:
            self.batch=[i for i in self.batch if self.items[i]['steps']<STEPS]
            if not self.batch and pending:
                first=min(r['arrival_s'] for r in pending)
                trigger=first if self.period==0 else math.ceil((first-1e-10)/self.period)*self.period
                if t+1e-10>=trigger:self.batch=[r['id'] for r in pending]
            if self.batch:return 'fu',self.batch[:]
            if fl<FL_STEPS:return 'fl',[]
        future=[r['arrival_s'] for r in self.items.values() if r['arrival_s']>t+1e-10]
        if pending and not self.parallel:
            first=min(r['arrival_s'] for r in pending)
            future.append(first if self.period==0 else math.ceil((first-1e-10)/self.period)*self.period)
        if future:return 'wait',min(future)
        assert not pending
        return 'done',[]
    def commit(self,ids,start,end):
        for i in ids:
            r=self.items[i];assert r['arrival_s']<=start+1e-10 and r['steps']<STEPS
            if r['first_service_s'] is None:r['first_service_s']=start
            r['steps']+=1
            if r['steps']==STEPS:r['completed_s']=end
    def metrics(self,t):
        seen=self.arrived(t);done=[r for r in seen if r['completed_s'] is not None and r['completed_s']<=t+1e-10]
        latency=[r['completed_s']-r['arrival_s'] for r in done]
        return dict(arrived=len(seen),procedure_completed=len(done),pending=len(seen)-len(done),
            completed_within_deadline=sum(r['completed_s']<=r['deadline_s']+1e-10 for r in done),
            overdue=sum(r['deadline_s']<t-1e-10 and (r['completed_s'] is None or r['completed_s']>r['deadline_s']+1e-10) for r in seen),
            mean_completion_latency_s=float(np.mean(latency)) if latency else None,
            max_completion_latency_s=max(latency) if latency else None,
            meaning='Five prescribed FU steps applied; not a certified deletion or a quality-pass count.')

def replay(requests,parallel,period,durations):
    q=Queue(requests,parallel,period);t=0.;fl=0;steps=0;occupied=0.;at_horizon=None
    while steps<2000:
        kind,ids=q.decision(t,fl)
        if kind=='done':break
        if kind=='wait':
            if at_horizon is None and t<=HORIZON<ids:at_horizon=dict(fl=fl,**q.metrics(HORIZON))
            t=ids;continue
        duration=durations[kind];end=t+duration
        if at_horizon is None and t<=HORIZON<end-1e-10:at_horizon=dict(fl=fl,**q.metrics(HORIZON))
        if kind in ['fl','joint'] and fl<FL_STEPS:fl+=1
        q.commit(ids,t,end);t=end;occupied+=duration;steps+=1
        if at_horizon is None and abs(t-HORIZON)<1e-10:at_horizon=dict(fl=fl,**q.metrics(t))
    assert steps<2000
    return dict(elapsed_s=t,occupied_s=occupied,fl_updates=fl,transactions=steps,
                request_metrics=q.metrics(t),at_common_horizon=at_horizon or dict(fl=fl,**q.metrics(t)))

def self_check():
    cohort=[list(range(100*u,100*u+90)) if u<5 else [] for u in range(10)]
    req=make_requests(cohort)
    d=dict(fl=FRAME_S,joint=FRAME_S,fu=FRAME_S)
    a=replay(req,True,0,d);b=replay(req,False,0,d)
    assert a['fl_updates']==b['fl_updates']==FL_STEPS
    assert a['request_metrics']['procedure_completed']==b['request_metrics']['procedure_completed']==45
    assert a['occupied_s']<b['occupied_s']
    q=Queue(req,False,5);assert q.decision(-FRAME_S,FL_STEPS)[0]=='wait'
    q=Queue(req,False,0);kind,ids=q.decision(0,0);assert kind=='fu'
    q.commit(ids,0,FRAME_S);assert all(q.items[i]['steps']==1 for i in ids)
    assert q.metrics(0)['procedure_completed']==0
    return dict(causal_arrival=True,no_future_completion=True,exact_service_steps=True,
        separate_pauses_FL_upload=True,joint_progresses_both=True,immediate_control=True)
