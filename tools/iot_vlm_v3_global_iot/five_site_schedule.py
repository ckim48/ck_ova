"""Five service regions with 36 adaptation FL updates each; no dropped FU."""
import copy
from collections import Counter


def prepare_requests(original,geo):
    assert len(geo['stops'])==5 and len(geo['FL_release_s'])==180
    result=[]
    for request in original:
        burst=geo['request_bursts'][request['burst']]
        completed=sum(t<burst['arrival_s']-1e-9 for t in geo['FL_release_s'])
        site=burst['site'];assert site*36<=completed<(site+1)*36
        item=copy.deepcopy(request)
        item.update(release_after_adaptation_FL=completed,release_before_total_FL_round=21+completed,
                    mission_site=site,old_mission_arrival_s=burst['arrival_s'],old_mission_deadline_s=burst['arrival_s']+30.,
                    old_service_arrival_s=request['arrival_s'],old_service_deadline_s=request['deadline_s'],
                    arrival_s=None,deadline_s=None)
        result.append(item)
    assert len(result)==45
    return result


def check_schedule(base,requests,geo):
    results={}
    for name,parallel in [('joint',True),('sequential',False)]:
        for duration in [.02048,1.]:
            q=base.RoundQueue(requests) if parallel else base.SerialRoundQueue(requests,False,5)
            fl=0;t=occupied=0.;counts=Counter();by_site=Counter()
            for _ in range(1000):
                if fl==180 and q.metrics(t)['procedure_completed']==45:break
                site=min(fl//36,4);stop=geo['stops'][site]
                t=max(t,stop['start_s']);q.sync(fl,t,occupied)
                kind,ids=q.decision(t,fl)
                if kind=='wait':t=ids;continue
                assert kind in ['joint','fl','fu']
                assert all(q.items[i]['mission_site']==site for i in ids)
                end=t+duration;assert end<=stop['end_s']+1e-9
                q.commit(ids,t,end);counts[kind]+=1
                if kind in ['joint','fl']:fl+=1;by_site[site]+=1
                occupied+=duration;t=end
            else:raise AssertionError('Schedule failed to terminate')
            assert fl==180 and all(v['steps']==5 for v in q.items.values())
            assert [by_site[i] for i in range(5)]==[36]*5
            if parallel:assert counts==Counter(joint=45,fl=135)
            else:assert counts==Counter(fl=180,fu=45)
            results[name+'_'+str(duration)]=dict(passed=True,total_FL=200,FU_requests=45,FU_steps=225,
                                               FL_updates_by_site=[by_site[i] for i in range(5)],transactions=dict(counts))
    return results
