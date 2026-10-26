import unittest
from comparison_schedule import SerialRoundQueue, replay, FRAME_S
from round_schedule import prepare_requests, replay as proposed_replay

class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.geo=dict(stops=[dict(start_s=t,end_s=t+60) for t in [0,150,300,520]])
        self.geo['FL_release_s']=[s['start_s']+1.2*j for s in self.geo['stops'] for j in range(45)]
        self.geo['request_bursts']=[dict(site=s,arrival_s=self.geo['stops'][s]['start_s']+o) for s,o in [(0,5),(0,25),(0,45),(1,10),(1,35),(2,10),(2,35),(3,10),(3,35)]]
        raw=[dict(id=f'b{b}_u{u}',burst=b,user=u,images=[b*10+u],required_steps=5,arrival_s=b*.4,deadline_s=b*.4+.4) for b in range(9) for u in range(5)]
        self.requests=prepare_requests(raw,self.geo)

    def test_same_work_but_serial_requires_more_transmissions(self):
        serial=replay(self.requests,self.geo,dict(fl=FRAME_S,fu=FRAME_S))
        proposed=proposed_replay(self.requests,self.geo)
        self.assertEqual(serial['counts'],dict(joint=0,fl=180,fu=45))
        self.assertEqual(serial['requests']['procedure_completed'],45)
        self.assertAlmostEqual(serial['occupied_uplink_s']-proposed['occupied_uplink_s'],45*FRAME_S)
        self.assertTrue(all(r['FL_after']==r['FL_before'] for r in serial['events'] if r['kind']=='fu'))

    def test_period_boundary_and_persistent_five_step_batch(self):
        q=SerialRoundQueue(self.requests)
        q.sync(4,.09,.09)
        self.assertEqual(q.pending(.09),[])
        q.sync(5,.11,.11)
        self.assertEqual(q.decision(.11,5),('fl',[]))
        q.sync(5,.2048,.2048)
        kind,ids=q.decision(.2048,5)
        self.assertEqual(kind,'fu');self.assertEqual(len(ids),5)
        for k in range(5):
            start=.2048+k*FRAME_S
            self.assertEqual(q.decision(start,5),('fu',ids))
            q.sync(5,start,start);q.commit(ids,start,start+FRAME_S)
        self.assertTrue(all(q.items[i]['completed_total_FL_round']==25 for i in ids))
        self.assertIsNone(q.metrics(.4)['completed_within_deadline'])

    def test_slow_radio_still_has_exact_FL_and_request_work(self):
        for d in [dict(fl=.6,fu=.4),dict(fl=1.,fu=.4)]:
            r=replay(self.requests,self.geo,d)
            self.assertEqual(r['completed_FL_rounds'],200)
            self.assertEqual(r['requests']['procedure_completed'],45)
            self.assertEqual(sum(len(x['requests']) for x in r['events']),225)

if __name__=='__main__':unittest.main()
