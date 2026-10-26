"""Finite continuous FL pilot: FU enters at fixed FL-round boundaries.

Mission v1's calendar is used only to fix each request's ordinal learning
position. It no longer gates FL. These are not the old time-arrival deadlines.
"""
import copy
import math


def prepare_requests(original, geo):
    result = []
    for r in original:
        b = geo['request_bursts'][r['burst']]
        site = b['site']
        old_time = b['arrival_s']
        # Count the old FL opportunities strictly preceding this event.
        completed = sum(at < old_time - 1e-9 for at in geo['FL_release_s'])
        assert site * 45 <= completed < (site + 1) * 45
        item = copy.deepcopy(r)
        item.update(release_after_adaptation_FL=completed,
                    release_before_total_FL_round=21 + completed,
                    mission_site=site, old_mission_arrival_s=old_time,
                    old_mission_deadline_s=old_time + 30.,
                    old_service_arrival_s=r['arrival_s'],
                    old_service_deadline_s=r['deadline_s'],
                    arrival_s=None, deadline_s=None)
        result.append(item)
    assert len(result) == 45
    return result


class RoundQueue:
    def __init__(self, requests):
        self.items = {r['id']: dict(copy.deepcopy(r), steps=0, first_service_s=None,
                                   completed_s=None, arrival_uplink_s=None,
                                   completed_uplink_s=None, completed_total_FL_round=None)
                      for r in requests}
        self.fl = 0
        self.occupied = 0.

    def sync(self, fl, mission_time, occupied):
        self.fl, self.occupied = fl, occupied
        for r in self.items.values():
            if r['arrival_s'] is None and r['release_after_adaptation_FL'] <= fl:
                r['arrival_s'] = mission_time
                r['arrival_uplink_s'] = occupied

    def arrived(self, t):
        return [r for r in self.items.values() if r['arrival_s'] is not None and r['arrival_s'] <= t + 1e-9]

    def pending(self, t):
        return [r for r in self.arrived(t) if r['steps'] < r['required_steps']]

    def decision(self, t, fl):
        ids = [r['id'] for r in self.pending(t)]
        if fl < 180:
            return ('joint' if ids else 'fl'), ids
        assert not ids, 'This fixed pilot must finish all FU within its 180 FL updates'
        assert all(r['steps'] == r['required_steps'] for r in self.items.values())
        return 'done', []

    def commit(self, ids, start, end):
        for rid in ids:
            r = self.items[rid]
            assert r['arrival_s'] is not None and r['arrival_s'] <= start + 1e-9
            assert r['steps'] < r['required_steps']
            if r['first_service_s'] is None:
                r['first_service_s'] = start
            r['steps'] += 1
            if r['steps'] == r['required_steps']:
                r.update(completed_s=end, completed_uplink_s=self.occupied + end - start,
                         completed_total_FL_round=21 + self.fl)

    def metrics(self, t):
        seen = self.arrived(t)
        done = [r for r in seen if r['completed_s'] is not None and r['completed_s'] <= t + 1e-9]
        lat = [r['completed_s'] - r['arrival_s'] for r in done]
        return dict(arrived=len(seen), procedure_completed=len(done), pending=len(seen)-len(done),
                    completed_within_deadline=None, overdue=None,
                    mean_completion_latency_s=sum(lat)/len(lat) if lat else None,
                    max_completion_latency_s=max(lat) if lat else None,
                    deadline_policy='Not evaluated: round-triggered requests replace calendar arrivals in this pilot.',
                    meaning='Five prescribed FU steps; not a certified deletion or quality-pass count.')


def replay(requests, geo, duration=.02048):
    q = RoundQueue(requests)
    t = occupied = 0.
    rows = []
    for fl in range(180):
        site = fl // 45
        t = max(t, geo['stops'][site]['start_s'])
        q.sync(fl, t, occupied)
        kind, ids = q.decision(t, fl)
        assert kind in ('fl', 'joint')
        assert (kind == 'joint') == bool(q.pending(t))
        assert all(q.items[rid]['mission_site'] == site for rid in ids)
        assert t + duration <= geo['stops'][site]['end_s'] + 1e-9
        q.commit(ids, t, t + duration)
        rows.append(dict(FL_round=21+fl, site=site, kind=kind, requests=ids, start_s=t, end_s=t+duration))
        t += duration
        occupied += duration
    assert q.metrics(t)['procedure_completed'] == 45
    assert all(r['steps'] == 5 for r in q.items.values())
    assert sum(r['kind'] == 'joint' for r in rows) == 45
    assert q.decision(t, 180) == ('done', [])
    return dict(passed=True, completed_FL_rounds=200, joint_transmissions=45,
                FL_only_transmissions=135, FU_only_transmissions=0, requests=q.metrics(t),
                occupied_uplink_s=occupied, last_update_mission_clock_s=t, transactions=rows,
                claim='CPU schedule validation only, not learned model performance.')
