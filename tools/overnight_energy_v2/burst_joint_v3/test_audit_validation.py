import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from audit_variants import validate_trial


class AuditValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.output = self.root/'new'; self.directory = self.output/'hybrid'; self.directory.mkdir(parents=True)
        self.plan = {'config':{'rounds':2}, 'source_sha256':{'example.py':'frozen'}}
        self.manifest = {'variant':'burst_joint_v3', 'algorithm':'coupled', 'device':'cuda',
            'transport':'hybrid', 'config':self.plan['config'], 'source_sha256':self.plan['source_sha256'], 'status':'completed'}
        self.schedule = [{'forget':[[1],[]]}, {'forget':[[],[]]}]
        self.rows = [{'round':1,'newly_deleted':1,'transmission':{'status':'feasible','energy_bound_j':.1},'cumulative_energy_bound_j':.1},
                     {'round':2,'newly_deleted':0,'transmission':{'status':'feasible','energy_bound_j':.2},'cumulative_energy_bound_j':.3}]
        baseline = self.root/'results/comparisons_seed0_solver_v2/coupled_hybrid'; baseline.mkdir(parents=True)
        (baseline/'schedule.json').write_text(json.dumps(self.schedule))
        (self.directory/'schedule.json').write_text(json.dumps(self.schedule))
        (self.directory/'variant_provenance.json').write_text(json.dumps(dict(mode='hybrid',**self.plan)))
        self.eq = SimpleNamespace(digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),
            read_rounds=lambda p:[json.loads(s) for s in (Path(p)/'rounds.jsonl').read_text().splitlines()])
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.directory/'manifest.json').write_text(json.dumps(self.manifest))
        (self.directory/'rounds.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))

    def validate(self):
        return validate_trial(self.root,self.output,'hybrid',self.plan,self.eq)

    def test_completed_valid_trial(self):
        self.assertEqual(self.validate()[1],'completed')

    def test_unhandled_request_is_not_complete(self):
        self.rows[0]['newly_deleted']=0; self.save()
        self.assertEqual(self.validate()[1],'incomplete_request_delivery')

    def test_provenance_and_energy_tampering_are_rejected(self):
        self.manifest['source_sha256']={}; self.save()
        self.assertEqual(self.validate()[1],'incompatible_variant_provenance')
        self.manifest['source_sha256']=self.plan['source_sha256']; self.rows[-1]['cumulative_energy_bound_j']=.1; self.save()
        self.assertEqual(self.validate()[1],'invalid_energy_total')


if __name__ == '__main__':
    unittest.main()
