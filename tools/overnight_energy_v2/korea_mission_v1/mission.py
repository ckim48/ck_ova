"""Actual OSM geometry, declared synthetic inspection mission and service clock."""
from pathlib import Path
import json, math, csv, hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/korea_mission_v1'
MAP=ROOT/'data/korea_mission_daejeon_v1'
R=6371008.8;LAT0=36.37;LON0=127.388

def xy(lat,lon):
    return np.array([R*math.cos(math.radians(LAT0))*math.radians(lon-LON0),R*math.radians(lat-LAT0)])

def latlon(p):
    return [LAT0+math.degrees(p[1]/R),LON0+math.degrees(p[0]/(R*math.cos(math.radians(LAT0))))]

def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=True))

def positions_at(t,spec):
    stops=spec['stops']
    for i,s in enumerate(stops):
        if s['start_s']<=t<=s['end_s']:return np.array(s['positions_xy_m']),i,'inspection_hover'
        if i and stops[i-1]['end_s']<t<s['start_s']:
            a=stops[i-1];f=(t-a['end_s'])/(s['start_s']-a['end_s'])
            return (1-f)*np.array(a['positions_xy_m'])+f*np.array(s['positions_xy_m']),None,'travel'
    last=stops[-1]
    if last['end_s']<t<spec['mission_end_s']:
        f=(t-last['end_s'])/(spec['mission_end_s']-last['end_s'])
        return (1-f)*np.array(last['positions_xy_m'])+f*np.array(stops[0]['positions_xy_m']),None,'return'
    return np.array(stops[0]['positions_xy_m']),0,'base_drain'

def window_at(t,spec):
    for i,s in enumerate(spec['stops']):
        if t<s['start_s']-1e-9:return None,s['start_s']
        if t<s['end_s']-1e-9:return i,s['end_s']
    if t<spec['mission_end_s']-1e-9:return None,spec['mission_end_s']
    return 0,float('inf')

def next_window(t,spec):
    for s in spec['stops']:
        if s['start_s']>t+1e-9:return s['start_s']
    return max(t,spec['mission_end_s'])

def requests_on_mission(requests,spec):
    result=[]
    for r in requests:
        q=dict(r);b=spec['request_bursts'][r['burst']]
        q.update(arrival_s=b['arrival_s'],deadline_s=b['arrival_s']+30.,mission_site=b['site'])
        result.append(q)
    assert len(result)==45
    return result

def decide(q,t,fl,spec):
    released=int(np.searchsorted(spec['FL_release_s'],t+1e-9,side='right'))
    ready=fl<released
    kind,ids=q.decision(t,fl if ready else 180)
    if kind=='joint' and not ready:kind='fu'
    if kind in ['wait','done']:
        future=[x for x in spec['FL_release_s'] if x>t+1e-9]
        if kind=='wait':future.append(ids)
        if future:return 'wait',min(future)
        return 'done',[]
    assert kind=='fu' or ready
    return kind,ids

def replay(requests,spec,Queue,parallel,period,durations):
    q=Queue(requests,parallel,period);t=0.;fl=steps=0;occupied=0.
    while steps<3000:
        if fl==180 and q.metrics(t)['procedure_completed']==45:break
        site,end=window_at(t,spec)
        if site is None:t=end;continue
        kind,ids=decide(q,t,fl,spec)
        if kind=='wait':t=ids;continue
        if kind=='done':break
        duration=durations[kind]
        if t+duration>end+1e-9:t=next_window(t,spec);continue
        assert kind=='fu' or fl<int(np.searchsorted(spec['FL_release_s'],t+1e-9,side='right'))
        q.commit(ids,t,t+duration)
        if kind in ['fl','joint']:fl+=1
        t+=duration;occupied+=duration;steps+=1
    assert fl==180 and q.metrics(t)['procedure_completed']==45 and steps<3000
    return dict(completed_FL=20+fl,requests=q.metrics(t),completion_clock_s=t,uplink_s=occupied,transactions=steps)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'mission_spec.json').exists():return json.loads((OUT/'mission_spec.json').read_text())
    raw=(MAP/'osm_snapshot.json').read_bytes();osm=json.loads(raw)
    provenance=json.loads((MAP/'provenance.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==provenance['sha256']
    ways={e['id']:e for e in osm['elements'] if e['type']=='way'}
    specs=[(30614866,'Hanbat Arboretum West','vegetation inspection'),
           (30614864,'Hanbat Arboretum East','vegetation inspection'),
           (28889891,'Expo Bridge sector','bridge and riverbank inspection'),
           (29416234,'Gapcheon river sector','waterway inspection')]
    stops=[];previous=None;clock=0.;speed=5.;height=50.
    grid=np.array([[60*(u%5-2),80*(u//5-.5)] for u in range(10)])
    for i,(oid,name,task) in enumerate(specs):
        geom=ways[oid]['geometry']
        if i==3:
            pt=min(geom,key=lambda p:(p['lat']-36.373)**2+(.8*(p['lon']-127.380))**2)
            center=xy(pt['lat'],pt['lon']);rule='Actual Gapcheon way vertex nearest declared mission anchor36.373N127.380E'
        else:
            center=np.mean([xy(p['lat'],p['lon']) for p in geom],axis=0);rule='Mean of OSM way geometry vertices'
        angle=math.radians([0,30,60,90][i]);rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        points=center+[.8,1.1,1.4,1.][i]*(grid@rot.T)
        if previous is not None:clock+=float(np.linalg.norm(points-previous,axis=1).max())/speed
        stops.append(dict(site=i,name=name,task=task,osm_way_id=oid,anchor_rule=rule,
            center_latlon=latlon(center),positions_xy_m=points.tolist(),start_s=clock,end_s=clock+60.))
        clock+=60.;previous=points
    clock+=float(np.linalg.norm(previous-np.array(stops[0]['positions_xy_m']),axis=1).max())/speed
    releases=[s['start_s']+1.2*j for s in stops for j in range(45)]
    burstslots=[(0,5),(0,25),(0,45),(1,10),(1,35),(2,10),(2,35),(3,10),(3,35)]
    bursts=[dict(burst=i,site=s,arrival_s=stops[s]['start_s']+offset) for i,(s,offset) in enumerate(burstslots)]
    result=dict(version='korea_mission_v1',stops=stops,mission_end_s=clock,cruise_speed_m_s=speed,altitude_m=height,
        FL_release_s=releases,request_bursts=bursts,projection=dict(type='Local spherical tangent approximation',R_m=R,lat0=LAT0,lon0=LON0),
        geometry_provenance=provenance,source='https://www.openstreetmap.org/copyright',
        semantics='Actual Korean OSM coordinates, synthetic formation/mission/releases. RSITMD/RSICD are onboard observation replays, not geotagged Korean mission images. No physical flight measurements.',
        timing='Four60s hover inspection windows,45 FL opportunities/site spaced1.2s. Travel at<=5m/s. Communications only during hover. Periodic FU keeps original5*20.48ms policy. Local compute/recovery/control/downlink timing not modeled; mission-service clock is not measured end-to-end latency.',
        late_work='Common planned mission and return are fixed; unfinished FL/FU drains at the first site after return. Report completion at common mission horizon and full workload separately.',
        channel='Existing noise-normalized free-space link model, 1000/d for beta0=-60dB and noise=-90dBm; recompute from actual inter-UAV distances at each stop. Perfect CSI and hover-static channels; no measured fading/building obstruction.',
        energy='Cumulative TX plus calibrated comm/compute if available. Do not invent missing device calibration or flight/hover energy. Record travel distance,hover time and active work for later accounting.')
    channels=[];minimum_separation=float('inf')
    for s in stops:
        p=np.array(s['positions_xy_m']);d=np.linalg.norm(p[:,None]-p[None],axis=-1)
        assert d[np.nonzero(d)].min()>20
        minimum_separation=min(minimum_separation,float(d[np.nonzero(d)].min()))
        h=1000/np.maximum(d,1);np.fill_diagonal(h,0);channels.append(h)
    assert not np.allclose(channels[0],channels[1])
    # Exact minimum pair separation along each linear formation transition.
    flight_min=float('inf')
    for a,b in zip(stops,stops[1:]+stops[:1]):
        pa=np.array(a['positions_xy_m']);pb=np.array(b['positions_xy_m'])
        for u in range(10):
            for v in range(u):
                delta=pa[u]-pa[v];velocity=(pb[u]-pb[v])-delta
                fraction=float(np.clip(-delta@velocity/max(velocity@velocity,1e-30),0,1))
                flight_min=min(flight_min,float(np.linalg.norm(delta+fraction*velocity)))
    assert flight_min>20,flight_min
    np.save(OUT/'channels_by_site.npy',np.array(channels))
    save('mission_spec.json',result)
    ticks=sorted(set([0.,clock,*np.arange(0,clock,2).tolist(),*[x for s in stops for x in [s['start_s'],s['end_s']]]]))
    with (OUT/'trajectory.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['mission_time_s','uav','latitude','longitude','altitude_m','east_m','north_m','phase','site'])
        previous=None;last=None;distances=np.zeros(10)
        for t in ticks:
            ps,site,phase=positions_at(t,result)
            if previous is not None:
                lengths=np.linalg.norm(ps-previous,axis=1);assert max(lengths)/(t-last)<=speed+1e-7;distances+=lengths
            for u,p in enumerate(ps):w.writerow([t,u,*latlon(p),height,*p,phase,site])
            previous=ps;last=t
    save('motion_checks.json',dict(passed=True,maximum_speed_m_s=speed,per_uav_distance_m=distances.tolist(),minimum_hover_separation_m=minimum_separation,minimum_flight_separation_m=flight_min,hover_duration_per_uav_s=240.,mission_duration_s=clock))
    return result
