"""Recover exact already-measured checkpoints within a common650kJ budget.

Replay original continuation without changing training or its400s horizon.
Audit every original evaluation/energy/time and final parameter tensor.
"""
import argparse
from pathlib import Path
import time

import learning_time_budget as original
import proposed_pks_report as current

ROOT, read, write, sha = original.ROOT, original.read, original.write, original.sha
OUT = ROOT/'results/tgcn_energy_checkpoint_v1_20261028'
METHODS = current.METHODS
BUDGET_J = 650000.


def register():
    path = OUT/'protocol.json'
    if path.exists():
        p = read(path)
        for name, digest in p['source_sha256'].items():
            assert sha(ROOT/name) == digest, name
        return p
    sources = dict(original.register()['source_sha256'])
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    p = dict(registered_unix=time.time(), source_sha256=sources, methods=METHODS,
        budget_j=BUDGET_J, policy='Exact replay of the already-published retained-only continuation after fixed200FL/225FU. Same400s execution horizon, same learning rate, sampler, channel and work accounting; save the last completed model with actual cumulative active energy<=650000J. No quality-driven stopping or interpolation.',
        integrity='Every original curve point and final parameter tensor must be exactly reproduced before using intermediate checkpoints. Existing original runs and failed attempts remain untouched.',
        evaluation='All45 FU procedures complete at this energy budget. Subsequent work is retained-only FL; this is not a new arriving-deletion experiment or a forgetting guarantee.')
    write(path, p)
    return p


def run(dataset):
    register()
    clone = original.quality.infra.cached_clone(original.run, [
        ("   extra=0;cached={};current_cost=np.zeros(10);job_no=0;all_jobs=[]",
         "   assert energy<=ENERGY_BUDGET_J\n   base.core.save_theta(ad/'snapshot650.pt',theta,200)\n   write(ad/'snapshot650.json',evaluations[-1])\n   extra=0;cached={};current_cost=np.zeros(10);job_no=0;all_jobs=[]", 1),
        ("   assert extra<1000 and active>=max(BUDGETS)",
         "    if energy<=ENERGY_BUDGET_J:\n     base.core.save_theta(ad/'snapshot650.pt',theta,200+extra)\n     write(ad/'snapshot650.json',evaluations[-1])\n   assert extra<1000 and active>=max(BUDGETS)", 1)
        ], dict(OUT=OUT, METHODS=METHODS, register=register, ENERGY_BUDGET_J=BUDGET_J))
    clone(dataset)
    import torch
    checks = []
    for m in METHODS:
        old = original.OUT/(dataset.lower()+'_seed0')/m
        new = OUT/(dataset.lower()+'_seed0')/m
        a, b = read(old/'curve.json'), read(new/'curve.json')
        assert a == b, (dataset, m, 'all observed curve points')
        assert torch.equal(torch.load(old/'final.pt', weights_only=True)['theta'],
                           torch.load(new/'final.pt', weights_only=True)['theta'])
        expected = next(p for p in reversed(a) if p['energy_j'] <= BUDGET_J)
        assert read(new/'snapshot650.json') == expected
        checks.append(dict(dataset=dataset, method=m, FL_round=expected['FL_round'],
            energy_j=expected['energy_j'], active_s=expected['active_s'],
            retained_mR_percent=100*expected['evaluation']['retained']['mR'],
            original_curve_and_final_model_exact=True,
            checkpoint=str((new/'snapshot650.pt').relative_to(ROOT)), checkpoint_sha256=sha(new/'snapshot650.pt')))
    write(OUT/(dataset.lower()+'_checks.json'), dict(passed=True, rows=checks))
    print(dict(dataset=dataset, replay_exact=True, models=len(checks)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=current.DATASETS)
    args = parser.parse_args()
    for ds in [args.dataset] if args.dataset else current.DATASETS:
        run(ds)
