"""Collection + actual service + immediate departure; no minimum service dwell."""
import copy
import numpy as np
from iot_energy import base as energy

class MissionClock:
    def __init__(self,geometry,deployment):
        self.geometry=geometry;self.deployment=deployment
        self.sites=[];self.legs=[];self.waits=[]
        self.durations=[]
        stops=geometry['stops']
        for i,a in enumerate(stops):
            end=stops[i+1]['start_s'] if i<len(stops)-1 else geometry['mission_end_s']
            seconds=end-a['end_s'];assert seconds>0;self.durations.append(seconds)

    def close_site(self,t,q,fl):
        s=self.sites[-1];i=s['site']
        assert fl==(i+1)*36,(i,fl)
        region=[r for r in q.items.values() if r['mission_site']==i]
        assert all(r['arrival_s'] is not None and r['steps']==r['required_steps'] for r in region)
        s.update(service_end_s=t,departure_s=t,service_duration_s=t-s['service_start_s'],
                 hover_duration_s=t-s['arrival_s'],completed_requests=len(region),FL_updates=36)

    def enter(self,site,t,q,fl):
        if self.sites and self.sites[-1]['site']==site:return t
        assert site==len(self.sites)
        if self.sites:
            self.close_site(t,q,fl)
            leg=dict(from_site=site-1,to_site=site,start_s=t,duration_s=self.durations[site-1])
            t+=leg['duration_s'];leg['end_s']=t;self.legs.append(leg)
        else:assert site==0 and fl==0 and t==0
        seconds=self.deployment['sites'][site]['collection_s']
        self.sites.append(dict(site=site,arrival_s=t,collection_start_s=t,collection_end_s=t+seconds,
                               collection_duration_s=seconds,service_start_s=t+seconds))
        return t+seconds

    def wait(self,start,end):
        assert end>=start
        self.waits.append(dict(site=self.sites[-1]['site'],start_s=start,end_s=end,duration_s=end-start))

    @staticmethod
    def transaction_timing(compute,radio):
        _,times=energy.compute_cost(compute['device_macs'])
        _,broadcast=energy.communication(radio)
        seconds=float(times.max())
        return dict(compute_s=seconds,uplink_s=radio['uplink_s'],downlink_s=broadcast,
                    duration_s=seconds+radio['uplink_s']+broadcast,
                    policy='Synchronous max-device compute, then uplink, then aggregate broadcast; completion after broadcast.')

    def finish(self,t,q,fl):
        assert len(self.sites)==5 and fl==180
        self.close_site(t,q,fl)
        service_end=t
        self.legs.append(dict(from_site=4,to_site=0,start_s=t,end_s=t+self.durations[4],duration_s=self.durations[4]))
        t+=self.durations[4]
        assert np.isclose(t,sum(s['hover_duration_s'] for s in self.sites)+sum(l['duration_s'] for l in self.legs))
        return dict(policy='Depart as soon as collection, 36 local FL updates and all region FU steps finish; no 60-second floor.',
                    clock_scope='Mission clock includes collection, compute, uplink, downlink, queue waits, all flight legs and return. Common prefix is before takeoff.',
                    sites=self.sites,legs=self.legs,waits=self.waits,service_end_s=service_end,mission_end_s=t)


def queue_classes(base):
    class ClockMixin:
        last_uplink_s=0.
        def commit(self,ids,start,end):
            super().commit(ids,start,end)
            # The inherited queue assumes its clock contains only uplink.
            # Keep the occupied-uplink diagnostic separate from full time.
            for rid in ids:
                r=self.items[rid]
                if r['completed_s']==end:r['completed_uplink_s']=self.occupied+self.last_uplink_s
        def metrics(self,t):
            result=super().metrics(t)
            result['clock_scope']='Collection/compute/uplink/downlink/waits/flight included; completion after five FU steps and final model broadcast.'
            return result
    class JointQueue(ClockMixin,base.RoundQueue):pass
    class SerialQueue(ClockMixin,base.SerialRoundQueue):pass
    return JointQueue,SerialQueue
