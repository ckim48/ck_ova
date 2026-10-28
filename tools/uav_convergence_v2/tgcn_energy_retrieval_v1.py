"""Actual retrieval at the last completed model under the same650kJ budget."""
from pathlib import Path
import time

import tgcn_retrieval_examples_v1 as original
import tgcn_energy_checkpoint_v1 as checkpoints

ROOT, read, write, sha = original.ROOT, original.read, original.write, original.sha
OUT = ROOT/'results/tgcn_energy_retrieval_v1_20261028'
data_for = original.data_for


def budget_path(dataset, method):
    return checkpoints.OUT/(dataset.lower()+'_seed0')/method


def register():
    path = OUT/'protocol.json'
    if path.exists():
        p = read(path)
        for name, digest in p['source_sha256'].items():
            assert sha(ROOT/name) == digest, name
        return p
    p = dict(original.register())
    sources = dict(p['source_sha256'])
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    rows = []
    for ds in original.DATASETS:
        audit = read(checkpoints.OUT/(ds.lower()+'_checks.json'))
        assert audit['passed']
        rows += audit['rows']
        for m in original.METHODS:
            for name in ['snapshot650.pt', 'snapshot650.json']:
                f = budget_path(ds, m)/name
                sources[str(f.relative_to(ROOT))] = sha(f)
    p.update(registered_unix=time.time(), source_sha256=sources, budget_j=650000., models=rows,
        checkpoint='Last completed evaluated model within the same650kJ active-energy budget. These are exactly recovered checkpoints from the existing400s continuation, independently audited against every original curve point and final parameters. All45 prescribed FU procedures complete. FL round counts differ because actual completed work differs. Same native gallery and captions; no interpolation or qualitative-outcome-based checkpoint choice.',
        companion='The original fixed200-round examples and every-query rank table remain available separately. Do not conflate same-energy and same-round comparisons.')
    write(path, p)
    return p


run = checkpoints.original.quality.infra.cached_clone(original.run, [
    ("theta = base.core.load_theta(other/'final.pt', engine.device)",
     "theta = base.core.load_theta(budget_path(ds, m)/'snapshot650.pt', engine.device)", 1),
    ("expected = read(other/'service_completion.json')['evaluation']['retained']",
     "expected = read(budget_path(ds, m)/'snapshot650.json')['evaluation']['retained']", 1)
    ], dict(OUT=OUT, register=register, budget_path=budget_path))


if __name__ == '__main__':
    run()
