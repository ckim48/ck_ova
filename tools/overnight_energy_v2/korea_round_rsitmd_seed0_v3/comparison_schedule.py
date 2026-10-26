"""Same round arrivals as the completed pilot; original periodic serial service."""
from pathlib import Path
import math
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'korea_joint_seed0_pilot_v2'))
from round_schedule import RoundQueue

FRAME_S = 20480 / 1e6

class SerialRoundQueue(RoundQueue):
    def __init__(self, requests, parallel=False, period_frames=5):
        assert parallel is False
        super().__init__(requests)
        self.period = period_frames * FRAME_S
        self.batch = []

    def decision(self, t, fl):
        pending = self.pending(t)
        self.batch = [i for i in self.batch if self.items[i]['steps'] < self.items[i]['required_steps']]
        trigger = None
        if not self.batch and pending:
            first = min(r['arrival_s'] for r in pending)
            trigger = first if self.period == 0 else math.ceil((first - 1e-10) / self.period) * self.period
            if t + 1e-10 >= trigger:
                self.batch = [r['id'] for r in pending]
        if self.batch:
            return 'fu', self.batch[:]
        if fl < 180:
            return 'fl', []
        if pending:
            assert trigger is not None and trigger > t
            return 'wait', trigger
        assert all(r['steps'] == r['required_steps'] for r in self.items.values())
        return 'done', []

    def commit(self, ids, start, end):
        super().commit(ids, start, end)
        for rid in ids:
            if self.items[rid]['completed_s'] == end:
                self.items[rid]['completed_total_FL_round'] = 20 + self.fl

def replay(requests, geo, durations, period=5):
    q = SerialRoundQueue(requests, False, period)
    fl = 0
    t = occupied = 0.
    rows = []
    for _ in range(1000):
        if fl == 180 and q.metrics(t)['procedure_completed'] == 45:
            break
        site = min(fl // 45, 3)
        t = max(t, geo['stops'][site]['start_s'])
        q.sync(fl, t, occupied)
        kind, ids = q.decision(t, fl)
        if kind == 'wait':
            t = ids
            continue
        assert kind in ['fl', 'fu']
        assert (kind == 'fu') == bool(ids)
        end = t + durations[kind]
        assert end <= geo['stops'][site]['end_s'] + 1e-9, 'Work exceeds fixed site window'
        before = fl
        q.commit(ids, t, end)
        if kind == 'fl': fl += 1
        assert kind != 'fu' or fl == before
        rows.append(dict(kind=kind, FL_before=20+before, FL_after=20+fl, site=site, requests=ids, start_s=t, end_s=end))
        occupied += end - t
        t = end
    else:
        raise AssertionError('Scheduler failed to terminate')
    assert fl == 180 and all(r['steps'] == 5 for r in q.items.values())
    assert sum(r['kind'] == 'fl' for r in rows) == 180
    return dict(passed=True, completed_FL_rounds=200, counts={k:sum(r['kind']==k for r in rows) for k in ['joint','fl','fu']},
                occupied_uplink_s=occupied, requests=q.metrics(t), events=rows,
                scope='CPU schedule validation, not learned performance')
