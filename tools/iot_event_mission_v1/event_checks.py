"""Independent checks of service completion, movement and cost clocks."""
from pathlib import Path
import json,math
import numpy as np
from mission_clock import MissionClock,queue_classes
from iot_energy import base as energy
ROOT=Path(__file__).resolve().parents[2]
read=lambda path:json.loads(Path(path).read_text())

def preflight(base,requests,geometry,deployment):
    Joint,Serial=queue_classes(base);results=[]
    for parallel in [True,False]:
        for seconds in [.02,2.]:
            q=Joint(requests) if parallel else Serial(requests,False,5)
            clock=MissionClock(geometry,deployment);t=occupied=0.;fl=0;counts={'joint':0,'fl':0,'fu':0}
            for _ in range(1000):
                if fl==180 and q.metrics(t)['procedure_completed']==45:break
                site=min(fl//36,4);t=clock.enter(site,t,q,fl);q.sync(fl,t,occupied)
                kind,ids=q.decision(t,fl)
                if kind=='wait':clock.wait(t,ids);t=ids;continue
                q.last_uplink_s=.01;q.commit(ids,t,t+seconds);counts[kind]+=1
                if kind in ['fl','joint']:fl+=1
                t+=seconds;occupied+=.01
            else:raise AssertionError('Scheduler did not terminate')
            timeline=clock.finish(t,q,fl)
            assert all(r['steps']==5 for r in q.items.values())
            assert all(s['FL_updates']==36 for s in timeline['sites'])
            assert counts==({'joint':45,'fl':135,'fu':0} if parallel else {'joint':0,'fl':180,'fu':45})
            assert all((s['service_duration_s']<60 if seconds<1 else s['service_duration_s']>60) for s in timeline['sites'])
            assert sum(s['collection_duration_s'] for s in timeline['sites'])==50
            for r in q.items.values():assert r['completed_uplink_s']-r['arrival_uplink_s']<=.2
            results.append(dict(parallel=parallel,transaction_seconds=seconds,counts=counts,mission_end_s=timeline['mission_end_s'],passed=True))
    timing=MissionClock.transaction_timing(dict(device_macs=[12e9]+[0.]*9),dict(uplink_s=.02,energy_bound_j=.01))
    assert math.isclose(timing['compute_s'],1.) and math.isclose(timing['downlink_s'],.655872)
    assert math.isclose(timing['duration_s'],1.675872)
    return dict(passed=True,cases=results,compute_uplink_downlink_checked=True)

def validate(directory,parallel,geometry):
    directory=Path(directory);manifest=read(directory/'manifest.json')
    assert manifest['status']=='completed_200FL_and_requests' and manifest['completed_FL_rounds']==200
    tx=[read(p) for p in sorted((directory/'transactions').glob('*.json'))]
    work={r['transaction']:r for r in (read(p) for p in sorted((directory/'work_attempts').glob('*.json')))}
    requests=read(directory/'request_records.json');timeline=read(directory/'mission_timeline.json')
    counts={k:sum(t['kind']==k for t in tx) for k in ['joint','fl','fu']}
    assert counts==({'joint':45,'fl':135,'fu':0} if parallel else {'joint':0,'fl':180,'fu':45})
    assert len(requests)==45 and sum(len(t['FU_request_ids']) for t in tx)==225
    for r in requests:
        serving=[t for t in tx if r['id'] in t['FU_request_ids']]
        assert len(serving)==r['steps']==5
        assert math.isclose(serving[0]['start_s'],r['first_service_s'])
        assert math.isclose(serving[-1]['end_s'],r['completed_s'])
        assert math.isclose(sum(t['radio']['uplink_s'] for t in tx if t['end_s']<=r['completed_s']+1e-9),r['completed_uplink_s'])
    for s in timeline['sites']:
        records=[t for t in tx if t['site']==s['site']]
        assert sum(t['kind'] in ['joint','fl'] for t in records)==36
        assert math.isclose(records[0]['start_s'],s['service_start_s'])
        assert math.isclose(records[-1]['end_s'],s['departure_s'])
        waits=sum(w['duration_s'] for w in timeline['waits'] if w['site']==s['site'])
        assert math.isclose(sum(t['end_s']-t['start_s'] for t in records)+waits,s['service_duration_s'])
        assert math.isclose(s['hover_duration_s'],s['collection_duration_s']+s['service_duration_s'])
    for t in tx:
        a=work[t['sequence']];assert a['status']=='transmitted'
        expected=MissionClock.transaction_timing(a['compute'],t['radio'])
        assert math.isclose(t['end_s']-t['start_s'],expected['duration_s'],abs_tol=1e-9)
        assert math.isclose(t['timing']['downlink_s'],expected['downlink_s'])
        if t['kind']=='joint':
            assert a['FL_receiver_apply_s']==a['FU_receiver_apply_s']
            assert a['FL_clients_apply_s']==a['FU_clients_apply_s']==t['end_s']
        if t['kind']=='fu':assert a['FL_round']==a['FL_round_after'] and a['FL_vector_norm']==0
    for i,leg in enumerate(timeline['legs']):
        a=geometry['stops'][i]
        end=geometry['stops'][i+1]['start_s'] if i<4 else geometry['mission_end_s']
        assert math.isclose(leg['duration_s'],end-a['end_s'])
        assert math.isclose(leg['start_s'],timeline['sites'][i]['departure_s'])
        if i<4:assert math.isclose(leg['end_s'],timeline['sites'][i+1]['arrival_s'])
    assert math.isclose(timeline['mission_end_s'],sum(s['hover_duration_s'] for s in timeline['sites'])+sum(l['duration_s'] for l in timeline['legs']))
    return dict(passed=True,counts=counts,FL_rounds=200,FU_requests=45,FU_steps=225,clock_components_checked=True,immediate_departure_checked=True)
