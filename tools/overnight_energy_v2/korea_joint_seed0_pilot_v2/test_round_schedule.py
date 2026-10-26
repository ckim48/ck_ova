import unittest
from round_schedule import RoundQueue, prepare_requests, replay


class ScheduleTest(unittest.TestCase):
    def setUp(self):
        self.geo = dict(stops=[dict(start_s=s, end_s=s+60.) for s in (0., 150., 300., 520.)])
        self.geo['FL_release_s'] = [s['start_s']+1.2*j for s in self.geo['stops'] for j in range(45)]
        slots = [(0,5),(0,25),(0,45),(1,10),(1,35),(2,10),(2,35),(3,10),(3,35)]
        self.geo['request_bursts'] = [dict(site=s, arrival_s=self.geo['stops'][s]['start_s']+o) for s,o in slots]
        raw = [dict(id=f'b{b}_u{u}', burst=b, user=u, images=[b*10+u], required_steps=5,
                    arrival_s=b*.4, deadline_s=b*.4+.4) for b in range(9) for u in range(5)]
        self.requests = prepare_requests(raw, self.geo)

    def test_joint_progress_and_fixed_budget(self):
        result = replay(self.requests, self.geo)
        self.assertEqual(result['joint_transmissions'], 45)
        self.assertEqual(result['FL_only_transmissions'], 135)
        self.assertEqual(result['requests']['procedure_completed'], 45)
        self.assertIsNone(result['requests']['completed_within_deadline'])
        self.assertAlmostEqual(result['occupied_uplink_s'], 180*.02048)

    def test_arrival_is_causal_and_commit_shares_end(self):
        q = RoundQueue(self.requests)
        q.sync(4, .08, .08)
        self.assertEqual(q.decision(.08, 4), ('fl', []))
        q.sync(5, .1, .1)
        kind, ids = q.decision(.1, 5)
        self.assertEqual(kind, 'joint')
        self.assertEqual(len(ids), 5)
        self.assertEqual([q.items[i]['steps'] for i in ids], [0]*5)
        q.commit(ids, .1, .12048)
        self.assertEqual([q.items[i]['steps'] for i in ids], [1]*5)

    def test_unfinished_fu_cannot_be_reported_as_completed_budget(self):
        q = RoundQueue(self.requests)
        q.sync(180, 600., 3.)
        with self.assertRaises(AssertionError):
            q.decision(600., 180)


if __name__ == '__main__':
    unittest.main()
